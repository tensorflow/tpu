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
	"testing"

	"github.com/tensorflow/tpu/tools/ctpu/ctrl"
	"google.golang.org/api/compute/v1"
	"google.golang.org/api/tpu/v1alpha1"
)

func TestListFlockStatus(t *testing.T) {
	c := listCmd{}

	type flockStatusTest struct {
		flock    flock
		expected string
	}

	flockStatusTests := []flockStatusTest{
		flockStatusTest{
			flock: flock{
				vm:  &ctrl.GCEInstance{&compute.Instance{Status: "RUNNING"}},
				tpu: &ctrl.TPUInstance{&tpu.Node{State: "READY"}},
			},
			expected: "running",
		},
		flockStatusTest{
			flock: flock{
				vm: &ctrl.GCEInstance{&compute.Instance{Status: "STOPPED"}},
			},
			expected: "paused",
		},
		flockStatusTest{
			flock:    flock{},
			expected: "--",
		},
		flockStatusTest{
			flock: flock{
				vm:  &ctrl.GCEInstance{&compute.Instance{Status: "STOPPING"}},
				tpu: &ctrl.TPUInstance{&tpu.Node{State: "DELETING"}},
			},
			expected: "unknown",
		},
	}

	for _, test := range flockStatusTests {
		status := c.flockStatus(&test.flock)
		if status != test.expected {
			t.Errorf("c.flockStatus(%v) = %q, want: %q", test.flock, status, test.expected)
		}
	}
}
