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
	"context"
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

type tpuLongRunningOperation struct {
	cp *TPUCP
	op *tpu.Operation
}

func (o *tpuLongRunningOperation) LoopUntilComplete() error {
	if o.op.Error != nil {
		return errors.New(o.op.Error.Message)
	}
	for i := 0; i < tpuMaxLoops; i++ {
		time.Sleep(5 * time.Second) // Poll every 5 seconds
		op, err := o.cp.operations.Get(o.op.Name).Do()
		if err != nil {
			return err
		}
		if op.Error != nil {
			return fmt.Errorf("error retrieving TPU operation: %#v, op.Error: %#v", op, op.Error)
		}
		if op.Done {
			return nil
		}
		// Every 20 seconds
		if i%4 == 0 {
			log.Println("TPU operation still running...")
		}
	}
	return fmt.Errorf("TPU operation still pending after 15 minutes: %q", o.op.Name)
}

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
	config      *config.Config
	serviceMgmt *serviceManagementCP
}

func newTPUCP(config *config.Config, client *http.Client, serviceManagementCP *serviceManagementCP, userAgent string) (*TPUCP, error) {
	tpuService, err := tpu.New(client)
	if err != nil {
		return nil, err
	}
	tpuService.UserAgent = userAgent

	computeService, err := compute.New(client)
	if err != nil {
		return nil, err
	}
	computeService.UserAgent = userAgent

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

// IsPreemptible returns true if the Cloud TPU is a preemptible Cloud TPU, false otherwise.
func (i *TPUInstance) IsPreemptible() bool {
	if i.SchedulingConfig != nil {
		return i.SchedulingConfig.Preemptible
	}
	return false
}

// IsReserved returns true if the Cloud TPU is a reserved Cloud TPU, false otherwise.
func (i *TPUInstance) IsReserved() bool {
	if i.SchedulingConfig != nil {
		return i.SchedulingConfig.Reserved
	}
	return false
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

// OptionallyRetrieveInstance retrieves the Instance from the TPU control plane.
//
// If enableAPIIfRequired is false and the TPU API has not been enabled, it returns immediately and does not enable the API.
func (g *TPUCP) OptionallyRetrieveInstance(enableAPIIfRequired bool) (instance *TPUInstance, apiEnabled bool, err error) {
	node, err := g.nodes.Get(g.nodeName()).Do()
	googError, ok := err.(*googleapi.Error)
	if ok && googError != nil && googError.Code == 404 {
		return nil, true, nil
	}
	if ok && googError != nil && googError.Code == 403 {
		// Check to see if the TPU API hasn't yet been enabled
		enabled, err := g.serviceMgmt.checkIfEnabled(tpuServiceAPIName)
		if err != nil {
			return nil, false, fmt.Errorf("error encountered while determining if API has been enabled: %#v, underlying error returned from the TPU API: %#v", err, googError)
		}
		if !enabled {
			if !enableAPIIfRequired {
				return nil, false, nil
			}
			log.Printf("Enabling the TPU API (this may take a while)...")
			err = g.serviceMgmt.enableService(tpuServiceAPIName)
			if err != nil {
				return nil, false, err
			}
			log.Printf("Successfully enabled the TPU API.")
			// Retry getting the instance after enabling the API.
			return g.OptionallyRetrieveInstance(enableAPIIfRequired)
		}
	}
	if node == nil {
		return nil, true, nil
	}
	return &TPUInstance{node}, true, nil
}

// Instance retrieves the instance from the TPU control plane.
func (g *TPUCP) Instance() (*TPUInstance, error) {
	instance, _, err := g.OptionallyRetrieveInstance(true)
	return instance, err
}

// ListInstances lists all TPUs within a zone of the GCP project.
func (g *TPUCP) ListInstances() ([]*TPUInstance, error) {
	nodes, err := g.nodes.List(g.parentPath()).Do()
	if err != nil {
		return nil, err
	}
	if nodes.NextPageToken != "" {
		log.Printf("Warning: It's possible that not all Cloud TPUs are listed.")
	}
	instances := make([]*TPUInstance, len(nodes.Nodes))
	for i, node := range nodes.Nodes {
		instances[i] = &TPUInstance{node}
	}
	return instances, nil
}

// ListVersions retrieves all available TensorFlow versions that can be used to create a Cloud TPU.
func (g *TPUCP) ListVersions() ([]*tpu.TensorFlowVersion, error) {
	versions, err := g.versions.List(g.parentPath()).PageSize(100).Do()
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
	locations, err := g.locations.List(fmt.Sprintf("projects/%s", g.config.Project)).Do()
	if err != nil {
		return nil, err
	}

	if locations.NextPageToken != "" {
		log.Printf("Warning: not all available TPU locations retrieved.")
	}
	return locations.Locations, nil
}

// ListSizes retrieves all TPU sizes for the current configured location.
func (g *TPUCP) ListSizes() ([]*tpu.AcceleratorType, error) {
	types, err := g.locations.AcceleratorTypes.List(fmt.Sprintf("projects/%s/locations/%s", g.config.Project, g.config.Zone)).Do()
	if err != nil {
		return nil, err
	}
	if types.NextPageToken != "" {
		log.Printf("Warning: not all available TPU sizes retrieved.")
	}
	return types.AcceleratorTypes, nil
}

const tpuMaxLoops = 180 // 15 minutes in 5 second increments

func (g *TPUCP) parentPath() string {
	return fmt.Sprintf("projects/%s/locations/%s", g.config.Project, g.config.Zone)
}

var legacyNetwork = net.IPv4(10, 240, 0, 0)

func (g *TPUCP) selectCIDRBlock(routes []*compute.Route, cidrBlockSize uint, network string) (string, error) {
	cidrBlocks := make([]*net.IPNet, 0, len(routes))
	for _, i := range routes {
		// Filter out network ranges that are not peered with our GCP VPC Network.
		if !strings.HasSuffix(i.Network, network) {
			continue
		}
		_, ipNet, err := net.ParseCIDR(i.DestRange)
		if err != nil {
			return "", err
		}
		maskSize, _ := ipNet.Mask.Size()
		if maskSize < 8 {
			continue
		}
		if legacyNetwork.Equal(ipNet.IP) && maskSize <= 16 {
			return "", fmt.Errorf("Cloud TPUs cannot be used with legacy networks, please create a new GCP project")
		}
		if maskSize <= 16 && ipNet.Contains(net.IPv4(10, 240, 1, 1)) && ipNet.Contains(net.IPv4(10, 240, 250, 250)) {
			return "", fmt.Errorf("existing routing entries appear to entirely cover the IP-range ctpu uses")
		}
		cidrBlocks = append(cidrBlocks, ipNet)
	}

	fourthOctetIncrement := 1 << (32 - cidrBlockSize)

	// Select a random IP address.
	for thirdOctet := byte(1); thirdOctet < 255; thirdOctet++ {
	nextCandidate:
		for fourthOctetBase := 1; fourthOctetBase < 255; fourthOctetBase += fourthOctetIncrement {
			for candidateFourthOctet := fourthOctetBase; candidateFourthOctet < fourthOctetBase+fourthOctetIncrement; candidateFourthOctet += 2 {
				candidateIPAddress := net.IPv4(10, 240, thirdOctet, byte(candidateFourthOctet))
				for _, block := range cidrBlocks {
					if block.Contains(candidateIPAddress) {
						continue nextCandidate
					}
				}
			}
			candidateIPAddress := net.IPv4(10, 240, thirdOctet, byte(fourthOctetBase))
			_, newCidr, err := net.ParseCIDR(fmt.Sprintf("%s/%d", candidateIPAddress.String(), cidrBlockSize))
			if err != nil {
				return "", fmt.Errorf("error parsing constructed CIDR: %v", err)
			}
			split := strings.Split(newCidr.String(), "/")
			if len(split) != 2 {
				return "", fmt.Errorf("error parsing cidr block %q", newCidr.String())
			}
			return split[0], nil
		}
	}
	return "", errors.New("no available CIDR blocks found")
}

// TODO: handle cidr block sizes larger than 24 bits.
var tpuDeviceNetworkSizes = map[string]uint{
	"v2-8":    29,
	"v2-32":   29,
	"v2-128":  27,
	"v2-256":  26,
	"v2-512":  25,
	"v3-8":    29,
	"v3-32":   29,
	"v3-64":   28,
	"v3-128":  27,
	"v3-256":  26,
	"v3-512":  25,
	"v3-1024": 24,
	// "v3-2048": 24,  // TODO(saeta): Support full-size pods.
}

// cidrBlockSize returns the number of ones in the CIDR range, or an error.
func (g *TPUCP) cidrBlockSize(hardwareType string) (ones uint, err error) {
	cidrBits, present := tpuDeviceNetworkSizes[hardwareType]
	if !present {
		return 0, fmt.Errorf("unknown TPU device size %q", hardwareType)
	}
	return cidrBits, nil
}

// CreateInstance creates the Cloud TPU with an API call to the TPU control plane.
func (g *TPUCP) CreateInstance(ctx context.Context, version string, preemptible, reserved bool, hardwareType, network string) (LongRunningOperation, error) {
	routeItems := make([]*compute.Route, 0)
	err := g.compute.Routes.List(g.config.Project).Pages(ctx, func(routeList *compute.RouteList) error {
		routeItems = append(routeItems, routeList.Items...)
		return nil
	})
	if err != nil {
		return nil, err
	}

	cidrBlockSize, err := g.cidrBlockSize(hardwareType)
	if err != nil {
		return nil, err
	}
	cidrBlock, err := g.selectCIDRBlock(routeItems, cidrBlockSize, network)
	if err != nil {
		return nil, err
	}

	node := tpu.Node{
		AcceleratorType:   hardwareType,
		CidrBlock:         cidrBlock,
		Description:       "A Cloud TPU created with the ctpu tool.",
		TensorflowVersion: version,
		SchedulingConfig:  &tpu.SchedulingConfig{Preemptible: preemptible, Reserved: reserved},
		Network:           network,
	}
	req := g.nodes.Create(g.parentPath(), &node)
	op, err := req.NodeId(g.config.FlockName).Do()
	if err != nil {
		googErr, ok := err.(*googleapi.Error)
		if ok && googErr.Code == 429 {
			return nil, fmt.Errorf("TPU quota exceeded on project %q", g.config.Project)
		}
		return nil, err
	}
	return &tpuLongRunningOperation{g, op}, nil
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
	return fmt.Sprintf("projects/%s/locations/%s/nodes/%s", g.config.Project, g.config.Zone, g.config.FlockName)
}

// DeleteInstance deletes a previously created Cloud TPU with an API call to the TPU control plane.
func (g *TPUCP) DeleteInstance() (LongRunningOperation, error) {
	op, err := g.nodes.Delete(g.nodeName()).Do()
	if err != nil {
		return nil, err
	}
	if op.Error != nil {
		return nil, errors.New(op.Error.Message)
	}
	return &tpuLongRunningOperation{g, op}, nil
}
