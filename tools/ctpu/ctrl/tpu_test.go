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

	"github.com/tensorflow/tpu/tools/ctpu/config"
	"google.golang.org/api/compute/v1"
	"google.golang.org/api/tpu/v1alpha1"
)

func TestCreateParentPath(t *testing.T) {
	cfg := &config.Config{
		Project:   "testProject",
		Zone:      "us-central1-f",
		FlockName: "alice",
	}
	cp := TPUCP{config: cfg}

	expected := "projects/testProject/locations/us-central1-f"
	if cp.parentPath() != expected {
		t.Errorf("wrong parent path: got %s, want %s", cp.parentPath(), expected)
	}

	cfg.Project = "otherProject"
	cfg.Zone = "us-central1-c"
	expected = "projects/otherProject/locations/us-central1-c"
	if cp.parentPath() != expected {
		t.Errorf("wrong parent path: got %s, want %s", cp.parentPath(), expected)
	}
}

func TestNodeName(t *testing.T) {
	cfg := &config.Config{
		Project:   "testProject",
		Zone:      "us-central1-f",
		FlockName: "alice",
	}
	cp := TPUCP{config: cfg}

	expected := "projects/testProject/locations/us-central1-f/nodes/alice"
	if cp.nodeName() != expected {
		t.Errorf("wrong node name: got %s, want %s", cp.nodeName(), expected)
	}

	cfg.Project = "otherProject"
	cfg.Zone = "us-central1-c"
	cfg.FlockName = "bob"
	expected = "projects/otherProject/locations/us-central1-c/nodes/bob"
	if cp.nodeName() != expected {
		t.Errorf("wrong node name: got %s, want %s", cp.nodeName(), expected)
	}
}

func TestInstanceNodeName(t *testing.T) {
	name := "projects/testProject/locations/us-central1-f/nodes/alice"
	nodeName := "alice"
	i := &TPUInstance{&tpu.Node{Name: name}}
	if i.NodeName() != nodeName {
		t.Errorf("i.NodeName() = %q, want %q", i.NodeName(), nodeName)
	}
}

func TestCidrBlockSelection(t *testing.T) {
	type testCase struct {
		existingRoutes []string
		want           string
	}
	testCases := []testCase{{
		existingRoutes: []string{},
		want:           "10.240.1.0/29",
	}, {
		existingRoutes: []string{"10.250.1.0/29"},
		want:           "10.240.1.0/29",
	}, {
		existingRoutes: []string{"10.240.1.0/29"},
		want:           "10.240.1.8/29",
	}, {
		existingRoutes: []string{"10.240.1.0/29", "10.240.1.8/29"},
		want:           "10.240.1.16/29",
	}, {
		existingRoutes: []string{"10.240.1.0/29", "10.240.1.8/29", "10.240.2.8/29"},
		want:           "10.240.1.16/29",
	}, {
		existingRoutes: []string{"10.240.1.0/29", "10.240.1.8/29", "10.240.2.24/29"},
		want:           "10.240.1.16/29",
	}, {
		existingRoutes: []string{"10.148.0.0/20", "10.142.0.0/20", "10.240.1.0/29", "10.240.1.8/29", "10.240.2.24/29"},
		want:           "10.240.1.16/29",
	}, {
		existingRoutes: []string{"10.148.0.0/20", "10.142.0.0/20", "0.0.0.0/0", "10.240.1.0/29", "10.240.1.8/29", "10.240.2.24/29"},
		want:           "10.240.1.16/29",
	}}

	g := TPUCP{}

	for _, tt := range testCases {
		routes := make([]*compute.Route, len(tt.existingRoutes))
		for i, block := range tt.existingRoutes {
			routes[i] = &compute.Route{DestRange: block}
		}
		got, err := g.selectCidrBlock(routes)
		if err != nil {
			t.Fatalf("g.selectCidrBlock(%v) returned err: %v", tt.existingRoutes, err)
		}
		if got != tt.want {
			t.Errorf("g.selectCidrBlock(%v) = %q, want: %q", tt.existingRoutes, got, tt.want)
		}
	}
}

func TestCidrBlockErrorHandling(t *testing.T) {
	malformed := []*compute.Route{
		{DestRange: "garbage"},
	}

	g := TPUCP{}
	got, err := g.selectCidrBlock(malformed)
	if err == nil {
		t.Fatalf("g.selectCidrBlock(--malformed--) = %v, %v, want: non-nil error value", got, err)
	}
}

func TestCidrBlockLegacyNetwork(t *testing.T) {
	includesLegacy := []*compute.Route{
		{DestRange: "0.0.0.0/0"},
		{DestRange: "10.240.0.0/16"},
	}

	g := TPUCP{}
	got, err := g.selectCidrBlock(includesLegacy)
	if err == nil {
		t.Fatalf("g.selectCidrBlock(%#v) = %v, %v, want: non-nil error value", includesLegacy, got, err)
	}
}

func TestCidrBlockLargeNetblocks(t *testing.T) {
	largeNetBlocks := []string{"10.240.0.0/12", "10.224.0.0/11", "10.128.0.0/9"}

	for _, netblock := range largeNetBlocks {
		networks := []*compute.Route{
			{DestRange: "0.0.0.0/0"},
			{DestRange: netblock},
		}

		g := TPUCP{}
		got, err := g.selectCidrBlock(networks)
		if err == nil {
			t.Fatalf("g.selectCidrBlock(%q) = %v, %v, want: non-nil error value", netblock, got, err)
		}
	}
}
