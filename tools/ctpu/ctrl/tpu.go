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
	"net"
	"net/http"
	"strings"
	"time"

	"github.com/tensorflow/tpu/tools/ctpu/config"
	"google.golang.org/api/compute/v1"
	"google.golang.org/api/googleapi"
	"google.golang.org/api/tpu/v1alpha1"
)

const tpuServiceAPIName = "tpu.googleapis.com"

// TPUCP contains an abstract representation of the Cloud TPU control plane.
//
// It is intentionally small so that other packages in the ctpu tool can be effectively
// tested.
type TPUCP struct {
	nodes       *tpu.ProjectsLocationsNodesService
	operations  *tpu.ProjectsLocationsOperationsService
	versions    *tpu.ProjectsLocationsTensorflowVersionsService
	locations   *tpu.ProjectsLocationsService
	compute     *compute.Service
	config      config.Config
	serviceMgmt *serviceManagementCP
}

func newTPUCP(config config.Config, client *http.Client, serviceManagementCP *serviceManagementCP, ctpuVersion string) (*TPUCP, error) {
	tpuService, err := tpu.New(client)
	if err != nil {
		return nil, err
	}
	tpuService.UserAgent = "ctpu/" + ctpuVersion

	computeService, err := compute.New(client)
	if err != nil {
		return nil, err
	}
	computeService.UserAgent = "ctpu/" + ctpuVersion

	return &TPUCP{
		nodes:       tpu.NewProjectsLocationsNodesService(tpuService),
		operations:  tpu.NewProjectsLocationsOperationsService(tpuService),
		versions:    tpu.NewProjectsLocationsTensorflowVersionsService(tpuService),
		locations:   tpu.NewProjectsLocationsService(tpuService),
		compute:     computeService,
		config:      config,
		serviceMgmt: serviceManagementCP,
	}, nil
}

// TPUInstance represents the Cloud TPU within the flock.
type TPUInstance struct {
	*tpu.Node
}

// IsRunning returns true if the Cloud TPU is running, false otherwise.
func (i *TPUInstance) IsRunning() bool {
	// Workaround for b/69965805
	return i.State == "READY" || (i.State == "CREATING" && len(i.IpAddress) > 0)
}

// NodeName returns the flock name (the human-usable name) of the Cloud TPU
func (i *TPUInstance) NodeName() string {
	parts := strings.Split(i.Name, "/")
	if len(parts) != 6 {
		log.Printf("Error parsing TPU name: %q", i.Name)
		return "__________"
	}
	return parts[len(parts)-1]
}

// Instance retrieves the Instance from the TPU control plane.
func (g *TPUCP) Instance() (*TPUInstance, error) {
	node, err := g.nodes.Get(g.nodeName()).Do()
	googError, ok := err.(*googleapi.Error)
	if ok && googError != nil && googError.Code == 404 {
		return nil, nil
	}
	if ok && googError != nil && googError.Code == 403 {
		// Check to see if the TPU API hasn't yet been enabled
		enabled, err := g.serviceMgmt.checkIfEnabled(tpuServiceAPIName)
		if err != nil {
			return nil, fmt.Errorf("error encountered while determining if API has been enabled: %#v, underlying error returned from the TPU API: %#v", err, googError)
		}
		if !enabled {
			log.Printf("Enabling the TPU API (this may take a while)...")
			err = g.serviceMgmt.enableService(tpuServiceAPIName)
			if err != nil {
				return nil, err
			}
			log.Printf("Successfully enabled the TPU API.")
			// Retry getting the instance after enabling the API.
			return g.Instance()
		}
	}
	if node == nil {
		return nil, nil
	}
	return &TPUInstance{node}, nil
}

// ListInstances lists all TPUs within a zone of the GCP project.
func (g *TPUCP) ListInstances() ([]*TPUInstance, error) {
	nodes, err := g.nodes.List(g.parentPath()).Do()
	if err != nil {
		return nil, err
	}
	if nodes.NextPageToken != "" {
		log.Printf("Warning: not all Cloud TPU's may be listed.")
	}
	instances := make([]*TPUInstance, len(nodes.Nodes))
	for i, node := range nodes.Nodes {
		instances[i] = &TPUInstance{node}
	}
	return instances, nil
}

// ListVersions retrieves all available TensorFlow versions that can be used to create a Cloud TPU.
func (g *TPUCP) ListVersions() ([]*tpu.TensorFlowVersion, error) {
	versions, err := g.versions.List(g.parentPath()).Do()
	if err != nil {
		return nil, err
	}
	if versions.NextPageToken != "" {
		log.Printf("Warning: not all available TF versions retrieved.")
	}
	return versions.TensorflowVersions, nil
}

// ListLocations retrieves all locations where TPUs might be available.
func (g *TPUCP) ListLocations() ([]*tpu.Location, error) {
	locations, err := g.locations.List(fmt.Sprintf("projects/%s", g.config.Project())).Do()
	if err != nil {
		return nil, err
	}

	if locations.NextPageToken != "" {
		log.Printf("Warning: not all available TPU locations retrieved.")
	}
	return locations.Locations, nil
}

func (g *TPUCP) loopUntilOperationComplete(operation *tpu.Operation) error {
	if operation.Error != nil {
		return errors.New(operation.Error.Message)
	}
	for {
		time.Sleep(5 * time.Second) // Poll every 5 seconds
		op, err := g.operations.Get(operation.Name).Do()
		if err != nil {
			return err
		}
		if op.Error != nil {
			return fmt.Errorf("error retrieving TPU operation: %#v, op.Error: %#v", op, op.Error)
		}
		if op.Done {
			return nil
		}
	}
}

func (g *TPUCP) parentPath() string {
	return fmt.Sprintf("projects/%s/locations/%s", g.config.Project(), g.config.Zone())
}

func (g *TPUCP) selectCidrBlock(routes []*compute.Route) (string, error) {
	cidrBlocks := make([]*net.IPNet, 0, len(routes))
	for _, i := range routes {
		_, ipNet, err := net.ParseCIDR(i.DestRange)
		if err != nil {
			return "", err
		}
		maskSize, _ := ipNet.Mask.Size()
		if maskSize < 8 {
			continue
		}
		cidrBlocks = append(cidrBlocks, ipNet)
	}

	// Select a random IP address.
	for thirdOctet := byte(1); thirdOctet < 255; thirdOctet++ {
	nextCandidate:
		for fourthOctet := byte(1); fourthOctet < 255; fourthOctet += 8 {
			candidateIPAddress := net.IPv4(10, 240, thirdOctet, fourthOctet)
			for _, block := range cidrBlocks {
				if block.Contains(candidateIPAddress) {
					continue nextCandidate
				}
			}
			_, newCidr, err := net.ParseCIDR(fmt.Sprintf("%s/29", candidateIPAddress.String()))
			if err != nil {
				return "", fmt.Errorf("error parsing constructed CIDR: %v", err)
			}
			return newCidr.String(), nil
		}
	}
	return "", errors.New("no available CIDR blocks found")
}

// CreateInstance creates the Cloud TPU with an API call to the TPU control plane.
func (g *TPUCP) CreateInstance(version string) error {
	routes, err := g.compute.Routes.List(g.config.Project()).Do()
	if err != nil {
		return err
	}

	cidrBlock, err := g.selectCidrBlock(routes.Items)
	if err != nil {
		return err
	}

	// TODO(saeta): Make TF version configurable, and default to a stable version.
	node := tpu.Node{
		AcceleratorType:   "v2-8",
		CidrBlock:         cidrBlock,
		Description:       "A Cloud TPU created with the ctpu tool.",
		TensorflowVersion: version,
	}
	req := g.nodes.Create(g.parentPath(), &node)
	op, err := req.NodeId(g.config.FlockName()).Do()
	if err != nil {
		googErr, ok := err.(*googleapi.Error)
		if ok && googErr.Code == 429 {
			return fmt.Errorf("TPU quota exceeded on project %q", g.config.Project())
		}
		return err
	}

	return g.loopUntilOperationComplete(op)
}

// StartInstance starts a previously stopped Cloud TPU with an API call to the TPU control plane.
func (g *TPUCP) StartInstance() error {
	return errors.New("starting a TPU is unimplemented")
}

// StopInstance stops a previously started Cloud TPU with an API call to the TPU control plane.
func (g *TPUCP) StopInstance(waitForAsync bool) error {
	return errors.New("stopping a TPU is unimplemented")
}

func (g *TPUCP) nodeName() string {
	return fmt.Sprintf("projects/%s/locations/%s/nodes/%s", g.config.Project(), g.config.Zone(), g.config.FlockName())
}

// DeleteInstance deletes a previously created Cloud TPU with an API call to the TPU control plane.
func (g *TPUCP) DeleteInstance(waitForAsync bool) error {
	op, err := g.nodes.Delete(g.nodeName()).Do()
	if err != nil {
		return err
	}
	if !waitForAsync {
		return nil
	}
	return g.loopUntilOperationComplete(op)
}
