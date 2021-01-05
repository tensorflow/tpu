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
	"context"
	"testing"

	"github.com/tensorflow/tpu/tools/ctpu/config"
	"google.golang.org/api/compute/v1"
	"google.golang.org/api/tpu/v1alpha1"
)

func testUpWorkflow(t *testing.T, libs *testLibs, expectedGCEAction, expectedTPUAction string, aclServiceAccount bool) {
	t.Helper()
	c := upCmd{
		cfg:              libs.cfg,
		tpu:              libs.tpu,
		gce:              libs.gce,
		rmg:              libs.rmg,
		cli:              libs.cli,
		machineType:      "n1-standard-2",
		diskSizeGb:       250,
		skipConfirmation: true,
	}

	exit := c.Execute(context.Background(), nil)
	if exit != 0 {
		t.Fatalf("Exit code incorrect: %d", exit)
	}

	verifySingleOperation(t, libs.gce.OperationsPerformed, expectedGCEAction)
	verifySingleOperation(t, libs.tpu.OperationsPerformed, expectedTPUAction)
	if aclServiceAccount {
		if libs.rmg.callCount != 1 {
			t.Errorf("resourceMgmt not called correct number of times: got %d, want 1", libs.rmg.callCount)
		}
	}
}

func TestUpAlreadyRunning(t *testing.T) {
	libs := newTestLibs()
	libs.gce.instance = &compute.Instance{Status: "RUNNING"}
	libs.tpu.instance = &tpu.Node{State: "READY"}
	libs.tpu.SetTfVersions("1.6", "1.7")
	testUpWorkflow(t, libs, "", "", false)
}

func TestUpCreating(t *testing.T) {
	libs := newTestLibs()
	libs.tpu.postCreateInstance = &tpu.Node{State: "READY", ServiceAccount: "compute-123@gserviceaccounts.com"}
	libs.tpu.SetTfVersions("1.6", "1.7")
	testUpWorkflow(t, libs, "CREATE", "CREATE-1.7", true)
	if libs.gce.createRequest == nil {
		t.Fatalf("createRequest was nil")
	}
	if libs.gce.createRequest.ImageFamily != "tf-1-7" {
		t.Errorf("createRequest.ImageFamily = %q, want: %q", libs.gce.createRequest.ImageFamily, "tf-1-7")
	}
	if libs.gce.createRequest.MachineType != "n1-standard-2" {
		t.Errorf("createRequest.MachineType = %q, want: %q", libs.gce.createRequest.MachineType, "n1-standard-2")
	}
	if libs.gce.createRequest.DiskSizeGb != 250 {
		t.Errorf("createRequest.ImageFamily = %d, want: %d", libs.gce.createRequest.DiskSizeGb, 250)
	}
}

func TestUpStarting(t *testing.T) {
	libs := newTestLibs()
	libs.gce.instance = &compute.Instance{Status: "STOPPING"}
	libs.tpu.postCreateInstance = &tpu.Node{State: "READY", ServiceAccount: "compute-123@gserviceaccounts.com"}
	libs.tpu.SetTfVersions("1.6")
	testUpWorkflow(t, libs, "START", "CREATE-1.6", true)
}

func TestUpAPIDisabled(t *testing.T) {
	libs := newTestLibs()
	libs.tpu.apiDisabled = true
	libs.tpu.postCreateInstance = &tpu.Node{State: "READY", ServiceAccount: "compute-123@gserviceaccounts.com"}
	libs.tpu.SetTfVersions("1.6", "1.7")
	testUpWorkflow(t, libs, "CREATE", "CREATE-1.7", true)
	if libs.gce.createRequest == nil {
		t.Fatalf("createRequest was nil")
	}
	if libs.gce.createRequest.ImageFamily != "tf-1-7" {
		t.Errorf("createRequest.ImageFamily = %q, want: %q", libs.gce.createRequest.ImageFamily, "tf-1-7")
	}
	if libs.gce.createRequest.MachineType != "n1-standard-2" {
		t.Errorf("createRequest.MachineType = %q, want: %q", libs.gce.createRequest.MachineType, "n1-standard-2")
	}
	if libs.gce.createRequest.DiskSizeGb != 250 {
		t.Errorf("createRequest.ImageFamily = %d, want: %d", libs.gce.createRequest.DiskSizeGb, 250)
	}
}

func TestUpMissingGcloud(t *testing.T) {
	libs := newTestLibs()
	libs.cli.isInstalled = false
	c := upCmd{
		cfg: libs.cfg,
		tpu: libs.tpu,
		gce: libs.gce,
		rmg: libs.rmg,
		cli: libs.cli,
	}
	exit := c.Execute(context.Background(), nil)
	if exit != 1 {
		t.Fatalf("Exit code incorrect, got: %d, want: 1", exit)
	}
}

func TestUpNoAvailableTfVersions(t *testing.T) {
	libs := newTestLibs()
	libs.tpu.postCreateInstance = &tpu.Node{State: "READY", ServiceAccount: "compute-123@gserviceaccounts.com"}
	libs.tpu.SetTfVersions("nightly")
	c := upCmd{
		cfg:              libs.cfg,
		tpu:              libs.tpu,
		gce:              libs.gce,
		rmg:              libs.rmg,
		cli:              libs.cli,
		skipConfirmation: true,
	}
	exit := c.Execute(context.Background(), nil)
	if exit != 1 {
		t.Errorf("Exit code incorrect, got: %d, want: 1", exit)
	}
}

func TestUpNoAvailableTfVersionsSetViaFlag(t *testing.T) {
	libs := newTestLibs()
	libs.tpu.postCreateInstance = &tpu.Node{State: "READY", ServiceAccount: "compute-123@gserviceaccounts.com"}
	libs.tpu.SetTfVersions("nightly")
	c := upCmd{
		cfg:              libs.cfg,
		tpu:              libs.tpu,
		gce:              libs.gce,
		rmg:              libs.rmg,
		cli:              libs.cli,
		tfVersion:        "nightly",
		skipConfirmation: true,
	}
	exit := c.Execute(context.Background(), nil)
	if exit != 0 {
		t.Errorf("Exit code incorrect, got: %d, want: 0", exit)
	}
}

func TestUpCreatingVmOnly(t *testing.T) {
	libs := newTestLibs()
	libs.tpu.SetTfVersions("1.6")
	c := upCmd{
		cfg:              libs.cfg,
		tpu:              libs.tpu,
		gce:              libs.gce,
		rmg:              libs.rmg,
		cli:              libs.cli,
		machineType:      "n1-standard-2",
		diskSizeGb:       250,
		skipConfirmation: true,
		vmOnly:           true,
	}

	exit := c.Execute(context.Background(), nil)
	if exit != 0 {
		t.Fatalf("Exit code incorrect: %d", exit)
	}

	verifySingleOperation(t, libs.gce.OperationsPerformed, "CREATE")
	verifySingleOperation(t, libs.tpu.OperationsPerformed, "")
	if libs.rmg.callCount != 0 {
		t.Errorf("resourceMgmt not called correct number of times: got %d, want 0", libs.rmg.callCount)
	}
}

func TestUpCreatingTPUOnly(t *testing.T) {
	libs := newTestLibs()
	libs.tpu.postCreateInstance = &tpu.Node{State: "READY", ServiceAccount: "compute-123@gserviceaccounts.com"}
	libs.tpu.SetTfVersions("1.6")
	c := upCmd{
		cfg:              libs.cfg,
		tpu:              libs.tpu,
		gce:              libs.gce,
		rmg:              libs.rmg,
		cli:              libs.cli,
		machineType:      "n1-standard-2",
		diskSizeGb:       250,
		skipConfirmation: true,
		tpuOnly:          true,
	}
	exit := c.Execute(context.Background(), nil)
	if exit != 0 {
		t.Fatalf("Exit code incorrect: %d", exit)
	}
	verifySingleOperation(t, libs.gce.OperationsPerformed, "")
	verifySingleOperation(t, libs.tpu.OperationsPerformed, "CREATE-1.6")
}

func TestUpImageFamily(t *testing.T) {
	testcases := []struct {
		tfVersion string
		want      string
	}{{
		tfVersion: "1.7",
		want:      "tf-1-7",
	}, {
		tfVersion: "1.6",
		want:      "tf-1-6",
	}, {
		tfVersion: "2.3.0",
		want:      "tf-2-3",
	}, {
		tfVersion: "2.4.0",
		want:      "tf-2-4-0",
	}, {
		tfVersion: "nightly",
		want:      "tf-nightly",
	}, {
		tfVersion: "nightly-20180203",
		want:      "",
	}, {
		tfVersion: "",
		want:      "",
	}, {
		tfVersion: "unparseable",
		want:      "",
	}}
	for _, testcase := range testcases {
		c := upCmd{
			tfVersion: testcase.tfVersion,
		}
		got, err := c.gceImageFamily()
		if testcase.want != "" && err != nil {
			t.Errorf("upCmd{tfVersion: %q}.gceImageFamily() = %v, want nil", testcase.tfVersion, err)
		}
		if testcase.want != got {
			t.Errorf("upCmd{tfVersion: %q}.gceImageFamily() = %q, want %q", testcase.tfVersion, got, testcase.want)
		}
	}
}

func TestCanoncalizeGCEImage(t *testing.T) {
	projectName := "test-project-name"

	testcases := []struct {
		input string
		want  string
	}{{
		input: "my-image",
		want:  "https://www.googleapis.com/compute/v1/projects/test-project-name/global/images/my-image",
	}, {
		input: "other-project/my-image",
		want:  "https://www.googleapis.com/compute/v1/projects/other-project/global/images/my-image",
	}, {
		input: "https://www.googleapis.com/compute/v1/projects/test-project-name/global/images/my-other-image",
		want:  "https://www.googleapis.com/compute/v1/projects/test-project-name/global/images/my-other-image",
	}, {
		input: "https://www.googleapis.com/compute/v1/projects/other-project-name/global/images/my-perfect-image",
		want:  "https://www.googleapis.com/compute/v1/projects/other-project-name/global/images/my-perfect-image",
	}, {
		input: "foo/bar/baz",
		want:  "",
	}}

	for _, tt := range testcases {
		t.Run(tt.input, func(t *testing.T) {
			cfg := &config.Config{Project: projectName}
			c := upCmd{cfg: cfg, gceImage: tt.input}
			err := c.canonicalizeGCEImage()
			if tt.want == "" {
				if err == nil {
					t.Errorf("Wanted non-nil error; got nil error.")
				}
			} else {
				if err != nil {
					t.Fatalf("c.canonicalizeGCEImage() = %#v, want: <nil>", err)
				}
				if tt.want != c.gceImage {
					t.Errorf("c.gceImage = %q, want: %q", c.gceImage, tt.want)
				}
			}
		})
	}
}
