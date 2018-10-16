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
	"os"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/tensorflow/tpu/tools/ctpu/config"
	"google.golang.org/api/tpu/v1alpha1"
)

func TestMakeExecCommandNoForwarding(t *testing.T) {
	cfg := &config.Config{
		FlockName: "testuser123",
		Project:   "testProject",
		Zone:      "us-central1-q",
	}
	cli := GCloudCLI{cfg}
	cmd := cli.makeExecCommand(false, false, nil)
	expected := []string{
		"gcloud",
		"compute",
		"ssh",
		"--zone",
		"us-central1-q",
		"--project",
		"testProject",
		"testuser123",
		"--",
		"-o",
		"SendEnv=TPU_NAME",
	}
	if !cmp.Equal(cmd, expected) {
		t.Errorf("incorrect command, got: %v, want: %v", cmd, expected)
	}
}

func TestMakeExecCommandForwardAgent(t *testing.T) {
	cfg := &config.Config{
		FlockName: "testuser321",
		Project:   "otherTestProject",
		Zone:      "us-central1-x",
	}
	cli := GCloudCLI{cfg}
	cmd := cli.makeExecCommand(false, true, nil)
	expected := []string{
		"gcloud",
		"compute",
		"ssh",
		"--zone",
		"us-central1-x",
		"--project",
		"otherTestProject",
		"testuser321",
		"--",
		"-o",
		"SendEnv=TPU_NAME",
		"-A",
	}
	if !cmp.Equal(cmd, expected) {
		t.Errorf("incorrect command, got: %v, want: %v", cmd, expected)
	}
}

func TestMakeExecCommandNoTPU(t *testing.T) {
	cfg := &config.Config{
		FlockName: "testuser321",
		Project:   "otherTestProject",
		Zone:      "us-central1-x",
	}
	cli := GCloudCLI{cfg}
	cmd := cli.makeExecCommand(true, false, nil)
	expected := []string{
		"gcloud",
		"compute",
		"ssh",
		"--zone",
		"us-central1-x",
		"--project",
		"otherTestProject",
		"testuser321",
		"--",
		"-o",
		"SendEnv=TPU_NAME",
		"-L",
		"6006:localhost:6006",
		"-L",
		"8888:localhost:8888",
	}
	if !cmp.Equal(cmd, expected) {
		t.Errorf("incorrect command, got: %v, want: %v", cmd, expected)
	}
}

func TestMakeExecCommandWithTPU(t *testing.T) {
	cfg := &config.Config{
		FlockName: "testuser321",
		Project:   "otherTestProject",
		Zone:      "us-central1-x",
	}
	tpu := &TPUInstance{&tpu.Node{
		NetworkEndpoints: []*tpu.NetworkEndpoint{
			{
				IpAddress: "10.123.210.25",
			},
		},
	}}
	cli := GCloudCLI{cfg}
	cmd := cli.makeExecCommand(true, false, tpu)
	expected := []string{
		"gcloud",
		"compute",
		"ssh",
		"--zone",
		"us-central1-x",
		"--project",
		"otherTestProject",
		"testuser321",
		"--",
		"-o",
		"SendEnv=TPU_NAME",
		"-L",
		"6006:localhost:6006",
		"-L",
		"8888:localhost:8888",
		"-L",
		"8470:10.123.210.25:8470",
		"-L",
		"8466:10.123.210.25:8466",
	}
	if !cmp.Equal(cmd, expected) {
		t.Errorf("incorrect command, got: %v, want: %v", cmd, expected)
	}
}

func TestMakeExecCommandWithTPUAndAgentForwarding(t *testing.T) {
	cfg := &config.Config{
		FlockName: "testuser321",
		Project:   "otherTestProject",
		Zone:      "us-central1-x",
	}
	tpu := &TPUInstance{&tpu.Node{
		NetworkEndpoints: []*tpu.NetworkEndpoint{
			{
				IpAddress: "10.123.210.25",
			},
		},
	}}
	cli := GCloudCLI{cfg}
	cmd := cli.makeExecCommand(true, true, tpu)
	expected := []string{
		"gcloud",
		"compute",
		"ssh",
		"--zone",
		"us-central1-x",
		"--project",
		"otherTestProject",
		"testuser321",
		"--",
		"-o",
		"SendEnv=TPU_NAME",
		"-A",
		"-L",
		"6006:localhost:6006",
		"-L",
		"8888:localhost:8888",
		"-L",
		"8470:10.123.210.25:8470",
		"-L",
		"8466:10.123.210.25:8466",
	}
	if !cmp.Equal(cmd, expected) {
		t.Errorf("incorrect command, got: %v, want: %v", cmd, expected)
	}
}

func TestMakeExecCommandInDevshell(t *testing.T) {
	cfg := &config.Config{
		FlockName:   "testuser321",
		Project:     "otherTestProject",
		Zone:        "us-central1-x",
		Environment: "devshell",
	}
	tpu := &TPUInstance{&tpu.Node{
		NetworkEndpoints: []*tpu.NetworkEndpoint{
			{
				IpAddress: "10.123.210.25",
			},
		},
	}}
	cli := GCloudCLI{cfg}
	cmd := cli.makeExecCommand(true, true, tpu)
	expected := []string{
		"gcloud",
		"compute",
		"ssh",
		"--zone",
		"us-central1-x",
		"--project",
		"otherTestProject",
		"testuser321",
		"--",
		"-o",
		"SendEnv=TPU_NAME",
		"-L",
		"8080:localhost:6006",
		"-L",
		"8081:localhost:8888",
		"-y",
	}
	if !cmp.Equal(cmd, expected) {
		t.Errorf("incorrect command, got: %v, want: %v", cmd, expected)
	}
}

func TestMakeEnviron(t *testing.T) {
	env := os.Environ()
	want := append(env, "TPU_NAME=my_flock_name")
	cfg := &config.Config{FlockName: "my_flock_name"}
	cli := GCloudCLI{cfg}
	got := cli.MakeEnviron()
	if !cmp.Equal(want, got) {
		t.Errorf("cli.MakeEnviron() = %v, want: %v", got, want)
	}
}
