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

func testPauseWorkflow(t *testing.T, libs *testLibs, expectedGCEAction, expectedTPUAction string) {
	t.Helper()
	c := pauseCmd{}

	exit := c.Execute(context.Background(), nil, libs.libs)
	if exit != 0 {
		t.Fatalf("Exit code incorrect: %d", exit)
	}

	verifySingleOperation(t, libs.testGCECP().OperationsPerformed, expectedGCEAction)
	verifySingleOperation(t, libs.testTPUCP().OperationsPerformed, expectedTPUAction)

	if libs.testRmg().callCount != 0 {
		t.Errorf("AddTPUUserAgent was called: %d", libs.testRmg().callCount)
	}
}

func TestPauseNotExistent(t *testing.T) {
	libs := newTestLibs()
	testPauseWorkflow(t, libs, "", "")
}

func TestPauseNotRunning(t *testing.T) {
	libs := newTestLibs()
	libs.testGCECP().instance = &compute.Instance{Status: "STOPPING"}
	libs.testTPUCP().instance = &tpu.Node{State: "CREATING"}
	testPauseWorkflow(t, libs, "", "DELETE")
}

func TestPause(t *testing.T) {
	libs := newTestLibs()
	libs.testGCECP().instance = &compute.Instance{Status: "RUNNING"}
	libs.testTPUCP().instance = &tpu.Node{State: "READY"}
	testPauseWorkflow(t, libs, "STOP", "DELETE")
}
