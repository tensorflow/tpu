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

package commands

import (
	"fmt"
	"sort"
	"testing"

	"context"
	"github.com/google/go-cmp/cmp"
	"github.com/tensorflow/tpu/tools/ctpu/config"
	"github.com/tensorflow/tpu/tools/ctpu/ctrl"
	"google.golang.org/api/compute/v1"
	"google.golang.org/api/tpu/v1alpha1"
)

type testLibs struct {
	cfg *config.Config
	gce *testGCECP
	tpu *testTPUCP
	cli *testGcloudCLI
	rmg *testResourceManagementCP
}

func newTestLibs() *testLibs {
	return &testLibs{
		cfg: &config.Config{
			FlockName:   "test",
			Environment: "gcloud",
			Project:     "test-project",
		},
		gce: &testGCECP{},
		tpu: &testTPUCP{},
		rmg: &testResourceManagementCP{},
		cli: &testGcloudCLI{true},
	}
}

type testLongRunningOperation struct {
	err error
}

func (t *testLongRunningOperation) LoopUntilComplete() error {
	if t.err != nil {
		return t.err
	}
	return nil
}

type testGCECP struct {
	instance            *compute.Instance
	OperationsPerformed []string
	instances           []compute.Instance

	createRequest *ctrl.GCECreateRequest

	apiDisabled bool
}

// Instance gets an instance associated with this control plane mock.
func (cp *testGCECP) Instance() (*ctrl.GCEInstance, error) {
	if cp.instance != nil {
		return &ctrl.GCEInstance{cp.instance}, nil
	}
	return nil, nil
}

func (cp *testGCECP) OptionallyRetrieveInstance(enableAPIIfRequired bool) (*ctrl.GCEInstance, bool, error) {
	if !enableAPIIfRequired && cp.apiDisabled {
		return nil, false, nil
	}
	cp.apiDisabled = false
	if cp.instance != nil {
		return &ctrl.GCEInstance{cp.instance}, true, nil
	}
	return nil, true, nil
}

// ListInstances lists all GCE instances in a given zone.
func (cp *testGCECP) ListInstances() ([]*ctrl.GCEInstance, error) {
	instances := make([]*ctrl.GCEInstance, len(cp.instances))
	for i, instance := range cp.instances {
		instances[i] = &ctrl.GCEInstance{&instance}
	}
	return instances, nil
}

// CreateInstance simulates creating the instance associated with this control plane mock.
func (cp *testGCECP) CreateInstance(req *ctrl.GCECreateRequest) (ctrl.LongRunningOperation, error) {
	if cp.createRequest == nil {
		cp.createRequest = req
	}
	cp.OperationsPerformed = append(cp.OperationsPerformed, "CREATE")
	return &testLongRunningOperation{}, nil
}

// StartInstance simulates starting the instance.
func (cp *testGCECP) StartInstance() (ctrl.LongRunningOperation, error) {
	cp.OperationsPerformed = append(cp.OperationsPerformed, "START")
	return &testLongRunningOperation{}, nil
}

// StopInstance simulates stopping the instance.
func (cp *testGCECP) StopInstance() (ctrl.LongRunningOperation, error) {
	cp.OperationsPerformed = append(cp.OperationsPerformed, "STOP")
	return &testLongRunningOperation{}, nil
}

// DeleteInstance simulates deleting the instance.
func (cp *testGCECP) DeleteInstance() (ctrl.LongRunningOperation, error) {
	cp.OperationsPerformed = append(cp.OperationsPerformed, "DELETE")
	return &testLongRunningOperation{}, nil
}

type testTPUCP struct {
	instance            *tpu.Node
	OperationsPerformed []string
	postCreateInstance  *tpu.Node
	instances           []tpu.Node

	tfVersions   []*tpu.TensorFlowVersion
	tpuLocations []*tpu.Location

	apiDisabled bool
}

func (cp *testTPUCP) Instance() (*ctrl.TPUInstance, error) {
	if cp.instance != nil {
		return &ctrl.TPUInstance{cp.instance}, nil
	}
	return nil, nil
}

func (cp *testTPUCP) OptionallyRetrieveInstance(enableAPIIfRequired bool) (*ctrl.TPUInstance, bool, error) {
	if !enableAPIIfRequired && cp.apiDisabled {
		return nil, false, nil
	}
	cp.apiDisabled = false
	if cp.instance != nil {
		return &ctrl.TPUInstance{cp.instance}, true, nil
	}
	return nil, true, nil
}

func (cp *testTPUCP) ListInstances() ([]*ctrl.TPUInstance, error) {
	instances := make([]*ctrl.TPUInstance, len(cp.instances))
	for i, instance := range cp.instances {
		instances[i] = &ctrl.TPUInstance{&instance}
	}
	return instances, nil
}

func (cp *testTPUCP) ListVersions() ([]*tpu.TensorFlowVersion, error) {
	return cp.tfVersions, nil
}

func (cp *testTPUCP) ListLocations() ([]*tpu.Location, error) {
	return cp.tpuLocations, nil
}

func (cp *testTPUCP) CreateInstance(ctx context.Context, version string, preemptible bool, hardwareType string) (ctrl.LongRunningOperation, error) {
	cp.OperationsPerformed = append(cp.OperationsPerformed, fmt.Sprintf("CREATE-%s", version))
	cp.instance = cp.postCreateInstance
	cp.postCreateInstance = nil
	return &testLongRunningOperation{}, nil
}

func (cp *testTPUCP) DeleteInstance() (ctrl.LongRunningOperation, error) {
	cp.OperationsPerformed = append(cp.OperationsPerformed, "DELETE")
	return &testLongRunningOperation{}, nil
}

func (cp *testTPUCP) SetTfVersions(versions ...string) {
	tfVersions := make([]*tpu.TensorFlowVersion, 0, len(versions))
	for _, version := range versions {
		tfVersion := &tpu.TensorFlowVersion{Version: version}
		tfVersions = append(tfVersions, tfVersion)
	}
	cp.tfVersions = tfVersions
}

func (cp *testTPUCP) SetLocations(locationIds ...string) {
	locations := make([]*tpu.Location, 0, len(locationIds))
	for _, loc := range locationIds {
		locations = append(locations, &tpu.Location{LocationId: loc})
	}
	cp.tpuLocations = locations
}

type testGcloudCLI struct {
	isInstalled bool
}

func (c *testGcloudCLI) IsGcloudInstalled() bool {
	return c.isInstalled
}

func (*testGcloudCLI) SSHToInstance(bool, bool, *ctrl.TPUInstance) error {
	return nil
}

func (testGcloudCLI) PrintInstallInstructions() {
	return
}

type testResourceManagementCP struct {
	callCount      int
	serviceAccount string
}

func (t *testResourceManagementCP) AddTPUUserAgent(tpuUserAgent string) error {
	t.callCount++
	t.serviceAccount = tpuUserAgent
	return nil
}

// verifySingleOperation is a helper used to test the operations the commands perform.
//
// It is often used in conjunction with the OperationsPerformed arrays of testTPUCP and testGCECP.
func verifySingleOperation(t *testing.T, operationsPerformed []string, expectedAction string) {
	t.Helper()
	if len(expectedAction) == 0 {
		if len(operationsPerformed) != 0 {
			t.Errorf("Operations performed: %#v, want: none", operationsPerformed)
		}
	} else {
		if len(operationsPerformed) != 1 || operationsPerformed[0] != expectedAction {
			t.Errorf("Operations performed: %#v, want: %s", operationsPerformed, expectedAction)
		}
	}
}

func TestParseVersion(t *testing.T) {
	testcases := []struct {
		version string
		want    parsedVersion
	}{{
		version: "1.6",
		want: parsedVersion{
			Major: 1,
			Minor: 6,
		},
	}, {
		version: "1.7",
		want: parsedVersion{
			Major: 1,
			Minor: 7,
		},
	}, {
		version: "nightly",
		want: parsedVersion{
			IsNightly: true,
		},
	}, {
		version: "nightly-20180205",
		want: parsedVersion{
			IsNightly: true,
			Modifier:  "-20180205",
		},
	}, {
		version: "unknown",
		want: parsedVersion{
			Modifier: "unknown",
		},
	}, {
		version: "1.7-RC3",
		want: parsedVersion{
			Major:    1,
			Minor:    7,
			Modifier: "-RC3",
		},
	}, {
		version: "1.7.2-RC0",
		want: parsedVersion{
			Major:    1,
			Minor:    7,
			Modifier: ".2-RC0",
		},
	}, {
		version: "test_version",
		want: parsedVersion{
			Modifier: "test_version",
		},
	}}
	for _, testcase := range testcases {
		got, err := parseVersion(testcase.version)
		if err != nil {
			t.Errorf("parseVersion(%q) = %#v", testcase.version, err)
		}
		if !cmp.Equal(got, testcase.want) {
			t.Errorf("parseVersion(%q) = %#v, want: %#v", testcase.version, got, testcase.want)
		}
	}
}

func TestSortingParsedVersions(t *testing.T) {
	testcases := []struct {
		versions []string
		want     []string
	}{{
		versions: []string{"1.7", "1.6"},
		want:     []string{"1.7", "1.6"},
	}, {
		versions: []string{"1.6", "1.7", "nightly"},
		want:     []string{"1.7", "1.6", "nightly"},
	}, {
		versions: []string{"1.6", "1.7", "nightly-20180901"},
		want:     []string{"1.7", "1.6", "nightly-20180901"},
	}, {
		versions: []string{"1.6", "1.7", "nightly", "nightly-20180901"},
		want:     []string{"1.7", "1.6", "nightly", "nightly-20180901"},
	}, {
		versions: []string{"1.6", "1.7", "nightly-20180901", "nightly"},
		want:     []string{"1.7", "1.6", "nightly", "nightly-20180901"},
	}, {
		versions: []string{"1.6", "1.7", "1.7RC1", "nightly"},
		want:     []string{"1.7", "1.7RC1", "1.6", "nightly"},
	}, {
		versions: []string{"1.7RC0", "1.7", "nightly"},
		want:     []string{"1.7", "1.7RC0", "nightly"},
	}, {
		versions: []string{"1.7RC0", "1.7RC1", "nightly"},
		want:     []string{"1.7RC1", "1.7RC0", "nightly"},
	}, {
		versions: []string{"2.1", "1.7", "nightly"},
		want:     []string{"2.1", "1.7", "nightly"},
	}, {
		versions: []string{"nightly", "2.1", "1.8"},
		want:     []string{"2.1", "1.8", "nightly"},
	}, {
		versions: []string{"nightly", "test_version", "2.1", "1.8"},
		want:     []string{"2.1", "1.8", "nightly", "test_version"},
	}, {
		versions: []string{"other_version", "nightly-20180901", "nightly", "nightly-20170101", "test_version"},
		want:     []string{"nightly", "nightly-20180901", "nightly-20170101", "other_version", "test_version"},
	}, {
		versions: []string{"test_version", "0.1", "other_version", "nightly", "nightly-20180901"},
		want:     []string{"0.1", "nightly", "nightly-20180901", "other_version", "test_version"},
	}}
	for _, testcase := range testcases {
		input := make([]parsedVersion, 0, len(testcase.versions))
		for _, version := range testcase.versions {
			parsed, err := parseVersion(version)
			if err != nil {
				t.Fatalf("Could not parse version %q", version)
			}
			if parsed.versionString() != version {
				t.Fatalf("parsedVersion(%q).versionString() = %q, want: %q", version, parsed.versionString(), version)
			}
			input = append(input, parsed)
		}
		sort.Sort(sortedParsedVersions(input))
		output := make([]string, 0, len(testcase.versions))
		for _, version := range input {
			output = append(output, version.versionString())
		}
		if !cmp.Equal(output, testcase.want) {
			t.Errorf("sort.Sort(sortedParsedVersions(%v)) = %#v, want: %#v", testcase.versions, output, testcase.want)
		}
	}
}

func TestLatestStableRelease(t *testing.T) {
	testcases := []struct {
		versions []string
		want     string
	}{{
		versions: []string{"1.7", "1.6"},
		want:     "1.7",
	}, {
		versions: []string{"1.6", "nightly"},
		want:     "1.6",
	}, {
		versions: []string{"1.7-RC0", "1.7", "nightly"},
		want:     "1.7",
	}, {
		versions: []string{"nightly", "nightly-20180201"},
		want:     "",
	}, {
		versions: []string{"1.7-RC0", "1.6", "1.5", "nightly", "nightly-20180201"},
		want:     "1.6",
	}}
	for _, testcase := range testcases {
		input := make([]parsedVersion, 0, len(testcase.versions))
		for _, version := range testcase.versions {
			parsed, err := parseVersion(version)
			if err != nil {
				t.Fatalf("Could not parse version %q", version)
			}
			if parsed.versionString() != version {
				t.Fatalf("parsedVersion(%q).versionString() = %q, want: %q", version, parsed.versionString(), version)
			}
			input = append(input, parsed)
		}
		output, err := sortedParsedVersions(input).LatestStableRelease()
		if len(testcase.want) > 0 && err != nil {
			t.Errorf("sortedParsedVersions(%#v).LatestStableRelease() = %v, want: nil", testcase.versions, err)
		}
		if testcase.want != output {
			t.Errorf("sortedParsedVersions(%#v).LatestStableRelease() = %q, want: %q", testcase.versions, output, testcase.want)
		}
	}
}
