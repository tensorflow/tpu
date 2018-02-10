// Copyright 2017 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ==============================================================================

package ctrl

import (
	"errors"
	"fmt"
	"log"
	"net/http"
	"regexp"
	"strings"
	"time"

	"github.com/tensorflow/tpu/tools/ctpu/config"
	"google.golang.org/api/compute/v1"
	"google.golang.org/api/googleapi"
)

const gceServiceAPIName = "compute.googleapis.com"

// GCECP contains an abstract representation of the GCE control plane.
//
// It is intentionally small so that other packages in the ctpu tool can be effectively
// tested.
type GCECP struct {
	computeService *compute.Service
	config         config.Config
	serviceMgmt    *serviceManagementCP
}

func newGCECP(config config.Config, client *http.Client, serviceManagementCP *serviceManagementCP, ctpuVersion string) (*GCECP, error) {
	computeService, err := compute.New(client)
	if err != nil {
		return nil, err
	}
	computeService.UserAgent = "ctpu/" + ctpuVersion
	return &GCECP{
		computeService: computeService,
		config:         config,
		serviceMgmt:    serviceManagementCP,
	}, nil
}

// GCEInstance represents the GCE instance within the flock.
type GCEInstance struct {
	*compute.Instance
}

// IsRunning returns true if the GCE instance is running, false otherwise.
func (i *GCEInstance) IsRunning() bool {
	return i.Status == "RUNNING"
}

// CanDelete returns true if the GCE intance can be deleted, false otherwise.
func (i *GCEInstance) CanDelete() bool {
	return i.IsRunning() || !strings.HasSuffix(i.Status, "ING")
}

// IsFlockVM returns true if this GCE VM appears to have been created by the ctpu tool.
func (i *GCEInstance) IsFlockVM() bool {
	_, ok := i.Labels["ctpu"]
	return ok
}

// Instance retrieves the Instance from the GCE control plane.
func (g *GCECP) Instance() (*GCEInstance, error) {
	instance, err := g.computeService.Instances.Get(g.config.Project(), g.config.Zone(), g.config.FlockName()).Do()
	googError, ok := err.(*googleapi.Error)
	if ok && googError != nil && googError.Code == 404 {
		return nil, nil
	}
	if ok && googError != nil && googError.Code == 403 {
		// Check to see if the GCE API hasn't yet been enabled.
		enabled, err := g.serviceMgmt.checkIfEnabled(gceServiceAPIName)
		if err != nil {
			return nil, fmt.Errorf("error encountered while determining if API has been enabled: %#v, underlying error returned from the GCE API: %#v", err, googError)
		}
		if !enabled {
			log.Printf("Enabling the GCE API (this may take a while)...")
			err = g.serviceMgmt.enableService(gceServiceAPIName)
			if err != nil {
				return nil, err
			}
			log.Printf("Successfully enabled the GCE API.")
			// Retry getting the instance after enabling the API.
			return g.Instance()
		}
	}
	if instance == nil {
		return nil, nil
	}
	return &GCEInstance{instance}, err
}

// ListInstances lists all GCE instances in a given zone.
func (g *GCECP) ListInstances() ([]*GCEInstance, error) {
	list, err := g.computeService.Instances.List(g.config.Project(), g.config.Zone()).Do()
	if err != nil {
		return []*GCEInstance{}, err
	}
	if list.NextPageToken != "" {
		log.Printf("Warning: not all GCE VM's may be listed.")
	}

	instances := make([]*GCEInstance, len(list.Items))
	for i, inst := range list.Items {
		instances[i] = &GCEInstance{inst}
	}

	return instances, nil
}

const gceMaxLoops = 120 // 10 minutes in 5 second increments

func (g *GCECP) loopUntilOperationComplete(operation *compute.Operation) error {
	if operation.Error != nil {
		return errors.New(operation.Error.Errors[0].Message)
	}
	for i := 0; i < gceMaxLoops; i++ {
		time.Sleep(5 * time.Second) // Poll every 5 seconds
		op, err := g.computeService.ZoneOperations.Get(g.config.Project(), g.config.Zone(), operation.Name).Do()
		if err != nil {
			return err
		}
		if op.Error != nil {
			return fmt.Errorf("error retrieving GCE zone operation: %#v", op)
		}
		if op.Status == "DONE" {
			return nil
		}
		// Every 30 seconds
		if i%6 == 0 {
			log.Println("GCE operation still running...")
		}
	}
	return fmt.Errorf("GCE operation still pending after 10 minutes: %q", operation.Name)
}

// GCECreateRequest captures all the configurable parameters involved in creating the GCE VM.
type GCECreateRequest struct {
	// ImageFamily is the name of the ever-green image family that should be used to create the GCE VM. It is resolved to the ImageName during the CreateInstance request.
	//
	// Exactly one of ImageFamily and ImageName should be non-empty.
	ImageFamily string

	// ImageName is the name of the image that should be used to create the GCE VM.
	//
	// Exactly one of ImageFamily and ImageName should be non-empty.
	ImageName string

	// TensorFlowVersion is the version of TensorFlow that is expected to be used.
	TensorFlowVersion string

	// MachineType is the GCE Machine type used when creating the instance.
	MachineType string

	// DiskSizeGb is the size the root volume should be sized to upon instance creation.
	DiskSizeGb int64
}

var conflictRegex = regexp.MustCompile("The resource 'projects/[^/]+/zones/([^/]+)/instances/([^']+)' already exists")

// CreateInstance creates the GCE instance with an API call to the GCE control plane.
func (g *GCECP) CreateInstance(request *GCECreateRequest) (err error) {
	if request.ImageName == "" && request.ImageFamily != "" {
		request.ImageName, err = g.resolveImageFamily(request.ImageFamily)
		if err != nil {
			return err
		}
	}
	if request.ImageName == "" {
		return fmt.Errorf("could not create GCE Instance without a base image")
	}
	instance := g.makeCreateInstance(request)
	op, err := g.computeService.Instances.Insert(g.config.Project(), g.config.Zone(), instance).Do()
	if err != nil {
		googErr, ok := err.(*googleapi.Error)
		if ok && googErr.Code == 409 {
			// Conflict, another GCE VM already exists.
			submatches := conflictRegex.FindStringSubmatch(googErr.Message)
			if len(submatches) == 3 && submatches[2] == g.config.FlockName() && submatches[1] != g.config.Zone() {
				return fmt.Errorf("while trying to create a GCE VM in zone %q, ctpu discovered another GCE VM of the same name (%q) has already been created in another zone (%q). Either use a new name (using the --name global flag), or use the other zone", g.config.Zone(), g.config.FlockName(), submatches[1])
			}
		}
		return err
	}
	return g.loopUntilOperationComplete(op)
}

// StartInstance starts a previously stopped GCE instance with an API call to the GCE control plane.
func (g *GCECP) StartInstance() error {
	op, err := g.computeService.Instances.Start(g.config.Project(), g.config.Zone(), g.config.FlockName()).Do()
	if err != nil {
		return err
	}
	return g.loopUntilOperationComplete(op)
}

// StopInstance stops a previously started GCE instance with an API call to the GCE control plane.
func (g *GCECP) StopInstance(waitForAsync bool) error {
	op, err := g.computeService.Instances.Stop(g.config.Project(), g.config.Zone(), g.config.FlockName()).Do()
	if err != nil {
		return err
	}
	if !waitForAsync {
		return nil
	}
	return g.loopUntilOperationComplete(op)
}

// DeleteInstance deletes a previously created GCE instance with an API call to the GCE control plane.
func (g *GCECP) DeleteInstance(waitForAsync bool) error {
	op, err := g.computeService.Instances.Delete(g.config.Project(), g.config.Zone(), g.config.FlockName()).Do()
	if err != nil {
		return err
	}
	if !waitForAsync {
		return nil
	}
	return g.loopUntilOperationComplete(op)
}

func (g *GCECP) resolveImageFamily(family string) (string, error) {
	image, err := g.computeService.Images.GetFromFamily("ml-images", family).Do()
	if err != nil {
		return "", nil
	}
	return image.SelfLink, nil
}

func (g *GCECP) makeCreateInstance(request *GCECreateRequest) *compute.Instance {
	flockName := g.config.FlockName()
	return &compute.Instance{
		Description: "TODO",
		Disks: []*compute.AttachedDisk{
			{
				Boot:       true,
				AutoDelete: true,
				InitializeParams: &compute.AttachedDiskInitializeParams{
					SourceImage: request.ImageName,
					DiskSizeGb:  request.DiskSizeGb,
				},
			},
		},
		NetworkInterfaces: []*compute.NetworkInterface{
			{
				Network: "global/networks/default",
				AccessConfigs: []*compute.AccessConfig{
					{
						Name: "External NAT",
						Type: "ONE_TO_ONE_NAT",
					},
				},
			},
		},
		ServiceAccounts: []*compute.ServiceAccount{
			{
				Email: "default",
				Scopes: []string{ // TODO(saeta): Make scopes configurable.
					"https://www.googleapis.com/auth/devstorage.read_write",
					"https://www.googleapis.com/auth/logging.write",
					"https://www.googleapis.com/auth/monitoring.write",
					"https://www.googleapis.com/auth/cloud-platform", // Ensure GCE has all potentially required permissions.
				},
			},
		},
		MachineType: fmt.Sprintf("zones/%s/machineTypes/%s", g.config.Zone(), request.MachineType),
		Name:        g.config.FlockName(),
		Labels: map[string]string{
			"ctpu": g.config.FlockName(),
		},
		Metadata: &compute.Metadata{
			Items: []*compute.MetadataItems{
				{
					Key:   "ctpu",
					Value: &flockName,
				},
			},
		},
	}
}
