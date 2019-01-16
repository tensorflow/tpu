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

// gceLongRunningOperation is returned when a control plane operation might take a while to complete.
type gceLongRunningOperation struct {
	cp *GCECP
	op *compute.Operation
}

func (o *gceLongRunningOperation) LoopUntilComplete() error {
	if o.op.Error != nil {
		return errors.New(o.op.Error.Errors[0].Message)
	}
	for i := 0; i < gceMaxLoops; i++ {
		time.Sleep(5 * time.Second) // Poll every 5 seconds
		op, err := o.cp.computeService.ZoneOperations.Get(o.cp.config.Project, o.cp.config.Zone, o.op.Name).Do()
		if err != nil {
			return err
		}
		if op.Error != nil {
			return fmt.Errorf("error retrieving Compute Engine zone operation: %#v", op)
		}
		if op.Status == "DONE" {
			return nil
		}
		// Every 30 seconds
		if i%6 == 0 {
			log.Println("Compute Engine operation still running...")
		}
	}
	return fmt.Errorf("Compute Engine operation still pending after 10 minutes: %q", o.op.Name)
}

// GCECP contains an abstract representation of the Compute Engine control plane.
//
// It is intentionally small so that other packages in the ctpu tool can be effectively
// tested.
type GCECP struct {
	computeService *compute.Service
	config         *config.Config
	serviceMgmt    *serviceManagementCP
}

func newGCECP(config *config.Config, client *http.Client, serviceManagementCP *serviceManagementCP, userAgent string) (*GCECP, error) {
	computeService, err := compute.New(client)
	if err != nil {
		return nil, err
	}
	computeService.UserAgent = userAgent
	return &GCECP{
		computeService: computeService,
		config:         config,
		serviceMgmt:    serviceManagementCP,
	}, nil
}

// GCEInstance represents the Compute Engine instance within the flock.
type GCEInstance struct {
	*compute.Instance
}

// IsRunning returns true if the Compute Engine instance is running, false otherwise.
func (i *GCEInstance) IsRunning() bool {
	return i.Status == "RUNNING"
}

// CanDelete returns true if the Compute Engine intance can be deleted, false otherwise.
func (i *GCEInstance) CanDelete() bool {
	return i.IsRunning() || !strings.HasSuffix(i.Status, "ING")
}

// IsFlockVM returns true if this Compute Engine VM appears to have been created by the ctpu tool.
func (i *GCEInstance) IsFlockVM() bool {
	_, ok := i.Labels["ctpu"]
	return ok
}

// OptionallyRetrieveInstance retrieves the instance from the Compute Engine control plane.
//
// If enableAPIIfRequired is false and the Compute Engine API has not been enabled, it returns immediately and does not enable the API.
func (g *GCECP) OptionallyRetrieveInstance(enableAPIIfRequired bool) (gceInstance *GCEInstance, apiEnabled bool, err error) {
	instance, err := g.computeService.Instances.Get(g.config.Project, g.config.Zone, g.config.FlockName).Do()
	googError, ok := err.(*googleapi.Error)
	if ok && googError != nil && googError.Code == 404 {
		return nil, true, nil
	}
	if ok && googError != nil && googError.Code == 403 {
		// Check to see if the Compute Engine API hasn't yet been enabled.
		enabled, err := g.serviceMgmt.checkIfEnabled(gceServiceAPIName)
		if err != nil {
			return nil, false, fmt.Errorf("error encountered while determining if API has been enabled: %#v, underlying error returned from the Compute Engine API: %#v", err, googError)
		}
		if !enabled {
			if !enableAPIIfRequired {
				return nil, false, nil
			}
			log.Printf("Enabling the Compute Engine API (this may take a while)...")
			err = g.serviceMgmt.enableService(gceServiceAPIName)
			if err != nil {
				return nil, false, err
			}
			log.Printf("Successfully enabled the Compute Engine API.")
			// Retry getting the instance after enabling the API.
			return g.OptionallyRetrieveInstance(enableAPIIfRequired)
		}
	}
	if instance == nil {
		return nil, true, nil
	}
	return &GCEInstance{instance}, true, err
}

// Instance retrieves the instance from the Compute Engine control plane.
func (g *GCECP) Instance() (*GCEInstance, error) {
	instance, _, err := g.OptionallyRetrieveInstance(true)
	return instance, err
}

// ListInstances lists all Compute Engine instances in a given zone.
func (g *GCECP) ListInstances() ([]*GCEInstance, error) {
	list, err := g.computeService.Instances.List(g.config.Project, g.config.Zone).Do()
	if err != nil {
		return []*GCEInstance{}, err
	}
	if list.NextPageToken != "" {
		log.Printf("Warning: It's possible that not all Compute Engine VMs are listed.")
	}

	instances := make([]*GCEInstance, len(list.Items))
	for i, inst := range list.Items {
		instances[i] = &GCEInstance{inst}
	}

	return instances, nil
}

const gceMaxLoops = 120 // 10 minutes in 5 second increments

// GCECreateRequest captures all the configurable parameters involved in creating the Compute Engine VM.
type GCECreateRequest struct {
	// ImageFamily is the name of the ever-green image family that should be used to create the Compute Engine VM. It is resolved to the ImageName during the CreateInstance request.
	//
	// Exactly one of ImageFamily and ImageName should be non-empty.
	ImageFamily string

	// ImageName is the name of the image that should be used to create the Compute Engine VM.
	//
	// Exactly one of ImageFamily and ImageName should be non-empty.
	ImageName string

	// TensorFlowVersion is the version of TensorFlow that is expected to be used.
	TensorFlowVersion string

	// MachineType is the Compute Engine machine type used when creating the instance.
	MachineType string

	// DiskSizeGb is the size the root volume should be sized to upon instance creation.
	DiskSizeGb int64

	// Preemptible is whether the Compute Engine VM runs in preemptible or not.
	Preemptible bool

	// Network is the network on which the Compute Engine VM should be created.
	Network string
}

var conflictRegex = regexp.MustCompile("The resource 'projects/[^/]+/zones/([^/]+)/instances/([^']+)' already exists")

// CreateInstance creates the Compute Engine instance with an API call to the Compute Engine control plane.
func (g *GCECP) CreateInstance(request *GCECreateRequest) (lro LongRunningOperation, err error) {
	if request.ImageName == "" && request.ImageFamily != "" {
		request.ImageName, err = g.resolveImageFamily(request.ImageFamily)
		if err != nil {
			return nil, err
		}
	}
	if request.ImageName == "" {
		return nil, fmt.Errorf("could not create Compute Engine instance without a base image")
	}
	instance := g.makeCreateInstance(request)
	op, err := g.computeService.Instances.Insert(g.config.Project, g.config.Zone, instance).Do()
	if err != nil {
		googErr, ok := err.(*googleapi.Error)
		if ok && googErr.Code == 409 {
			// Conflict, another Compute Engine VM already exists.
			submatches := conflictRegex.FindStringSubmatch(googErr.Message)
			if len(submatches) == 3 && submatches[2] == g.config.FlockName && submatches[1] != g.config.Zone {
				return nil, fmt.Errorf("while trying to create a Compute Engine VM in zone %q, ctpu discovered another Compute Engine VM of the same name (%q) has already been created in another zone (%q). Either use a new name (using the --name global flag), or use the other zone", g.config.Zone, g.config.FlockName, submatches[1])
			}
		}
		return nil, err
	}
	if op.Error != nil {
		return nil, errors.New(op.Error.Errors[0].Message)
	}
	return &gceLongRunningOperation{g, op}, nil
}

// StartInstance starts a previously stopped Compute Engine instance with an API call to the Compute Engine control plane.
func (g *GCECP) StartInstance() (LongRunningOperation, error) {
	op, err := g.computeService.Instances.Start(g.config.Project, g.config.Zone, g.config.FlockName).Do()
	if err != nil {
		return nil, err
	}
	if op.Error != nil {
		return nil, errors.New(op.Error.Errors[0].Message)
	}
	return &gceLongRunningOperation{g, op}, nil
}

// StopInstance stops a previously started Compute Engine instance with an API call to the Compute Engine control plane.
func (g *GCECP) StopInstance() (LongRunningOperation, error) {
	op, err := g.computeService.Instances.Stop(g.config.Project, g.config.Zone, g.config.FlockName).Do()
	if err != nil {
		return nil, err
	}
	if op.Error != nil {
		return nil, errors.New(op.Error.Errors[0].Message)
	}
	return &gceLongRunningOperation{g, op}, nil
}

// DeleteInstance deletes a previously created Compute Engine instance with an API call to the Compute Engine control plane.
func (g *GCECP) DeleteInstance() (LongRunningOperation, error) {
	op, err := g.computeService.Instances.Delete(g.config.Project, g.config.Zone, g.config.FlockName).Do()
	if err != nil {
		return nil, err
	}
	if op.Error != nil {
		return nil, errors.New(op.Error.Errors[0].Message)
	}
	return &gceLongRunningOperation{g, op}, nil
}

func (g *GCECP) resolveImageFamily(family string) (string, error) {
	image, err := g.computeService.Images.GetFromFamily("ml-images", family).Do()
	if err != nil {
		return "", nil
	}
	return image.SelfLink, nil
}

func (g *GCECP) makeCreateInstance(request *GCECreateRequest) *compute.Instance {
	flockName := g.config.FlockName
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
				Network: "global/networks/" + request.Network,
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
					"https://www.googleapis.com/auth/cloud-platform", // Ensure Compute Engine has all potentially required permissions.
				},
			},
		},
		MachineType: fmt.Sprintf("zones/%s/machineTypes/%s", g.config.Zone, request.MachineType),
		Name:        g.config.FlockName,
		Labels: map[string]string{
			"ctpu": g.config.FlockName,
		},
		Metadata: &compute.Metadata{
			Items: []*compute.MetadataItems{
				{
					Key:   "ctpu",
					Value: &flockName,
				},
			},
		},
		Scheduling: &compute.Scheduling{
			Preemptible: request.Preemptible,
		},
	}
}
