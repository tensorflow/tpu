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

	"context"
	"google.golang.org/api/compute/v1"
	"google.golang.org/api/tpu/v1alpha1"
)

func testUpWorkflow(t *testing.T, libs *testLibs, expectedGCEAction, expectedTPUAction string, aclServiceAccount bool) {
	t.Helper()
	c := upCmd{}

	exit := c.Execute(context.Background(), nil, libs.libs)
	if exit != 0 {
		t.Fatalf("Exit code incorrect: %d", exit)
	}

	verifySingleOperation(t, libs.testGCECP().OperationsPerformed, expectedGCEAction)
	verifySingleOperation(t, libs.testTPUCP().OperationsPerformed, expectedTPUAction)
	if aclServiceAccount {
		if libs.testRmg().callCount != 1 {
			t.Errorf("resourceMgmt not called correct number of times: got %d, want 1", libs.testRmg().callCount)
		}
	}
}

func TestUpAlreadyRunning(t *testing.T) {
	libs := newTestLibs()
	libs.testGCECP().instance = &compute.Instance{Status: "RUNNING"}
	libs.testTPUCP().instance = &tpu.Node{State: "READY"}
	libs.testTPUCP().SetTfVersions("1.6", "1.7")
	testUpWorkflow(t, libs, "", "", false)
}

func TestUpCreating(t *testing.T) {
	libs := newTestLibs()
	libs.testTPUCP().postCreateInstance = &tpu.Node{State: "READY", ServiceAccount: "compute-123@gserviceaccounts.com"}
	libs.testTPUCP().SetTfVersions("1.6", "1.7")
	testUpWorkflow(t, libs, "CREATE", "CREATE-1.7", true)
	if libs.testGCECP().createRequest == nil {
		t.Fatalf("createRequest was nil")
	}
	if libs.testGCECP().createRequest.ImageFamily != "tf-1-7" {
		t.Errorf("createRequest.ImageFamily = %q, want: %q", libs.testGCECP().createRequest.ImageFamily, "tf-1-7")
	}
}

func TestUpStarting(t *testing.T) {
	libs := newTestLibs()
	libs.testGCECP().instance = &compute.Instance{Status: "STOPPING"}
	libs.testTPUCP().postCreateInstance = &tpu.Node{State: "READY", ServiceAccount: "compute-123@gserviceaccounts.com"}
	libs.testTPUCP().SetTfVersions("1.6")
	testUpWorkflow(t, libs, "START", "CREATE-1.6", true)
}

func TestUpMissingGcloud(t *testing.T) {
	libs := newTestLibs()
	libs.testGcloudCLI().isInstalled = false
	c := upCmd{}
	exit := c.Execute(context.Background(), nil, libs.libs)
	if exit != 1 {
		t.Fatalf("Exit code incorrect, got: %d, want: 1", exit)
	}
}

func TestUpNoAvailableTfVersions(t *testing.T) {
	libs := newTestLibs()
	libs.testTPUCP().postCreateInstance = &tpu.Node{State: "READY", ServiceAccount: "compute-123@gserviceaccounts.com"}
	libs.testTPUCP().SetTfVersions("nightly")
	c := upCmd{}
	exit := c.Execute(context.Background(), nil, libs.libs)
	if exit != 1 {
		t.Errorf("Exit code incorrect, got: %d, want: 1", exit)
	}
}

func TestUpNoAvailableTfVersionsSetViaFlag(t *testing.T) {
	libs := newTestLibs()
	libs.testTPUCP().postCreateInstance = &tpu.Node{State: "READY", ServiceAccount: "compute-123@gserviceaccounts.com"}
	libs.testTPUCP().SetTfVersions("nightly")
	c := upCmd{
		tfVersion: "nightly",
	}
	exit := c.Execute(context.Background(), nil, libs.libs)
	if exit != 0 {
		t.Errorf("Exit code incorrect, got: %d, want: 0", exit)
	}
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
