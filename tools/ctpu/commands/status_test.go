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
	"time"

	"context"
	"github.com/fatih/color"
	"github.com/tensorflow/tpu/tools/ctpu/config"
	"github.com/tensorflow/tpu/tools/ctpu/ctrl"
	"google.golang.org/api/compute/v1"
	"google.golang.org/api/tpu/v1alpha1"
)

func TestRunnableStatus(t *testing.T) {
	color.NoColor = true
	s := statusCmd{}
	testCases := []struct {
		exists    bool
		isRunning bool
		status    string
		expected  string
	}{
		{false, false, "???", "--"},
		{true, false, "STOPPED", "STOPPED"},
		{true, false, "PROVISIONING", "PROVISIONING"},
		{true, true, "RUNNING", "RUNNING"},
	}

	for _, tt := range testCases {
		actual := s.runnableStatus(tt.exists, tt.isRunning, tt.status)
		if actual != tt.expected {
			t.Errorf("s.runnableStatus(%t, %t, %q) = %q, want %q", tt.exists, tt.isRunning, tt.status, actual, tt.expected)
		}
	}
}

func TestFlockStatus(t *testing.T) {
	color.NoColor = true
	s := statusCmd{}
	testCases := []struct {
		vm       *ctrl.GCEInstance
		tpu      *ctrl.TPUInstance
		expected string
	}{
		{nil, nil, "No instances currently exist."},
		{&ctrl.GCEInstance{&compute.Instance{Status: "RUNNING"}}, &ctrl.TPUInstance{&tpu.Node{State: "READY"}}, "Your cluster is running!"},
		{&ctrl.GCEInstance{&compute.Instance{Status: "STOPPING"}}, &ctrl.TPUInstance{&tpu.Node{State: "DELETING"}}, "Your cluster is in an unhealthy state."},
		{&ctrl.GCEInstance{&compute.Instance{Status: "STOPPED"}}, nil, "Your cluster is paused."},
	}

	for _, tt := range testCases {
		actual := s.flockStatus(tt.vm, tt.tpu)
		if actual != tt.expected {
			t.Errorf("s.flockStatus(%v, %v) = %q, want %q", tt.vm, tt.tpu, actual, tt.expected)
		}
	}
}

func TestStatusExecute(t *testing.T) {
	testCases := []struct {
		vm  *compute.Instance
		tpu *tpu.Node
	}{
		{nil, nil},
		{&compute.Instance{Status: "RUNNING"}, &tpu.Node{State: "READY", SchedulingConfig: &tpu.SchedulingConfig{}}},
		{&compute.Instance{Status: "STOPPING"}, &tpu.Node{State: "DELETING", SchedulingConfig: &tpu.SchedulingConfig{}}},
		{&compute.Instance{Status: "STOPPED"}, nil},
		{nil, &tpu.Node{State: "DELETING", SchedulingConfig: &tpu.SchedulingConfig{}}},
	}

	for _, tt := range testCases {
		for _, details := range []bool{false, true} {
			s := statusCmd{
				cfg: &config.Config{
					FlockName: "test-flock",
					Project:   "testProject",
				},
				tpu:     &testTPUCP{instance: tt.tpu},
				gce:     &testGCECP{instance: tt.vm},
				details: details,
			}

			got := s.Execute(context.Background(), nil)
			if got != 0 {
				t.Errorf("statusCmd{details: %t}.Execute(nil, nil, %v) = %d, want: 0", details, tt, got)
			}
		}
	}
}

func TestTimeFormatTest(t *testing.T) {
	testCases := []struct {
		createTime     time.Time
		expectedFormat string
	}{{
		createTime:     time.Now(),
		expectedFormat: "< 1 minute",
	}, {
		createTime:     time.Now().Add(time.Minute * -3),
		expectedFormat: "3m",
	}, {
		createTime:     time.Now().Add(time.Hour * -27).Add(time.Minute * -14),
		expectedFormat: "27h 14m",
	}, {
		createTime:     time.Now().Add(time.Minute * 3),
		expectedFormat: "--",
	}, {
		createTime:     time.Now().Add(time.Hour * -103).Add(time.Minute * -26),
		expectedFormat: "4d 7h",
	}}

	s := statusCmd{}
	for _, tt := range testCases {
		result := s.timeDelta(tt.createTime)
		if result != tt.expectedFormat {
			t.Errorf("s.timeDelta(%s) = %s; want: %s", tt.createTime, result, tt.expectedFormat)
		}
	}
}

func TestTimeParseFormat(t *testing.T) {
	utcMinus7 := time.FixedZone("PDT", -7*60*60) // -7:00

	testCases := []struct {
		input  string
		format string
		want   time.Time
	}{{
		input:  "2018-06-12T10:54:21.812-07:00",
		format: time.RFC3339,
		want:   time.Date(2018, 06, 12, 10, 54, 21, 812000000, utcMinus7),
	}, {
		input:  "2018-06-12T17:54:21.767342Z",
		format: time.RFC3339,
		want:   time.Date(2018, 06, 12, 17, 54, 21, 767342000, time.UTC),
	}}

	for _, tt := range testCases {
		got, err := time.Parse(tt.format, tt.input)
		if err != nil {
			t.Errorf("time.Parse(%q, %q) had an error; got: %v", tt.format, tt.input, err)
			continue
		}
		if !got.Equal(tt.want) {
			t.Errorf("time.Parse(%q, %q) = %v; want: %v", tt.format, tt.input, got, tt.want)
		}
	}
}
