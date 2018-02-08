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

	"github.com/google/go-cmp/cmp"
	"google.golang.org/api/tpu/v1alpha1"
)

func TestTpuLocationsSort(t *testing.T) {
	testcases := []struct {
		locations []string
		want      []string
	}{{
		locations: []string{"us-central1-f", "us-central1-c"},
		want:      []string{"us-central1-c", "us-central1-f"},
	}, {
		locations: []string{"us-central1-f", "us-central1-c", "us-central1-b"},
		want:      []string{"us-central1-b", "us-central1-c", "us-central1-f"},
	}, {
		locations: []string{"us-central1-b", "us-central1-c", "us-central1-f"},
		want:      []string{"us-central1-b", "us-central1-c", "us-central1-f"},
	}}
	for _, testcase := range testcases {
		input := make([]*tpu.Location, 0, len(testcase.locations))
		for _, locID := range testcase.locations {
			input = append(input, &tpu.Location{LocationId: locID})
		}
		sortLocations(input)
		output := make([]string, 0, len(testcase.locations))
		for _, loc := range input {
			output = append(output, loc.LocationId)
		}
		if !cmp.Equal(output, testcase.want) {
			t.Errorf("sort.Sort(byLocID(%#v)) = %#v, want: %#v", testcase.locations, output, testcase.want)
		}
	}
}
