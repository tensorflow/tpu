// Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
	"reflect"
	"testing"

	"google.golang.org/api/tpu/v1alpha1"
)

func TestTPUSizeSort(t *testing.T) {
	testcases := []struct {
		input []string
		want  []string
	}{{
		input: []string{"v2-8"},
		want:  []string{"v2-8"},
	}, {
		input: []string{"v2-8", "v3-8"},
		want:  []string{"v2-8", "v3-8"},
	}, {
		input: []string{"v3-8", "v2-8"},
		want:  []string{"v2-8", "v3-8"},
	}, {
		input: []string{"v3-8", "v2-8", "v3-32", "v2-64"},
		want:  []string{"v2-8", "v2-64", "v3-8", "v3-32"},
	}}

	for i, tt := range testcases {
		if len(tt.input) != len(tt.want) {
			t.Fatalf("Invalid test case: %d.", i)
		}
		input := make([]*tpu.AcceleratorType, len(tt.input))
		for i, name := range tt.input {
			input[i] = &tpu.AcceleratorType{Type: name}
		}
		want := make([]*tpu.AcceleratorType, len(tt.want))
		for i, name := range tt.want {
			want[i] = &tpu.AcceleratorType{Type: name}
		}
		sortTpuSizes(input)
		if !reflect.DeepEqual(input, want) {
			fmt.Printf("Failure on test case %d\n", i)
			for i, _ := range tt.want {
				fmt.Printf("%d: want: %q, got: %q\n", i, want[i].Type, input[i].Type)
			}
			t.Errorf("Input: %v, want: %v, got: %v", tt.input, want, input)
		}
	}
}
