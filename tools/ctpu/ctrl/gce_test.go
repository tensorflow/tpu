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
	"testing"

	"cmp"
	"github.com/tensorflow/tpu/tools/ctpu/config"
)

func TestMakeCreateInstance(t *testing.T) {
	cfg := config.TestConfig{
		ActiveConfigurationVal: "ctpu9",
		FlockNameVal:           "testFlock",
		IPRangeVal:             "10.240.0.0/16",
		ProjectVal:             "ctpu-test-project",
		ZoneVal:                "us-central1-c",
	}
	cp := GCECP{
		config: &cfg,
	}
	request := GCECreateRequest{
		ImageName:   "my_image_name",
		MachineType: "n1-standard-2",
		DiskSizeGb:  350,
	}
	i := cp.makeCreateInstance(&request)
	if i == nil {
		t.Fatal("instance was nil")
	}
	if i.Name != "testFlock" {
		t.Errorf("cp.makeCreateInstance(...).Name = %v, want %v", i.Name, "testFlock")
	}
	// TODO(saeta): Test description
	if len(i.Disks) != 1 {
		t.Fatalf("wrong disk size: got %d, want 1", len(i.Disks))
	}
	if i.Disks[0].InitializeParams.SourceImage != "my_image_name" {
		t.Errorf("i.Disks[0].InitializeParams.SourceImage = %s, want my_image_name", i.Disks[0].InitializeParams.SourceImage)
	}
	if i.Disks[0].InitializeParams.DiskSizeGb != 350 {
		t.Errorf("i.Disks[0].InitializeParams.DiskSizeGb = %d, want %d", i.Disks[0].InitializeParams.DiskSizeGb, 350)
	}
	if len(i.NetworkInterfaces) != 1 {
		t.Errorf("wrong number of network interfaces: got %d, want 1", len(i.NetworkInterfaces))
	}
	mt := "zones/us-central1-c/machineTypes/n1-standard-2"
	if i.MachineType != mt {
		t.Errorf("wrong machine type: got %s, want %s", i.MachineType, mt)
	}

	if i.Labels["ctpu"] != "testFlock" {
		t.Errorf("cp.makeCreateInstance(my_image_name).Labels[ctpu] = %s, want testFlock", i.Labels["ctpu"])
	}

	if i.Metadata == nil {
		t.Errorf("cp.makeCreateInstance(my_image_name).Metadata = nil, want non-nil")
	} else if len(i.Metadata.Items) != 1 {
		t.Errorf("len(cp.makeCreateInstance(my_image_name).Metadata) = %d, want 1", len(i.Metadata.Items))
	} else if i.Metadata.Items[0].Key != "ctpu" || *i.Metadata.Items[0].Value != "testFlock" {
		t.Errorf("cp.makeCreateInstance(my_image_name).Metadata[0] = %#v, want ctpu:testFlock", i.Metadata.Items[0])
	}

	gce := GCEInstance{i}
	if !gce.IsFlockVM() {
		t.Errorf("gce.IsFlockVM() = %v, want %v", gce.IsFlockVM(), true)
	}
}

func TestConflictRegex(t *testing.T) {
	testCases := []struct {
		input string
		want  []string
	}{{
		input: "malformed",
		want:  nil,
	}, {
		input: "",
		want:  nil,
	}, {
		input: "The resource 'projects/testProject/zones/us-central1-c/instances/example' already exists",
		want:  []string{"The resource 'projects/testProject/zones/us-central1-c/instances/example' already exists", "us-central1-c", "example"},
	}}

	for _, tt := range testCases {
		got := conflictRegex.FindStringSubmatch(tt.input)
		if !cmp.Equal(got, tt.want) {
			t.Errorf("conflictRegex.FindStringSubmatch(%q) = %q, want %q", tt.input, got, tt.want)
		}
	}
}
