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
	"fmt"
	"testing"

	"context"
	"github.com/google/subcommands"
	"github.com/tensorflow/tpu/tools/ctpu/config"
	"github.com/tensorflow/tpu/tools/ctpu/ctrl"
	"google.golang.org/api/tpu/v1alpha1"
)

type restartTestLRO struct {
	wait bool
}

func (r *restartTestLRO) LoopUntilComplete() error {
	r.wait = true
	return nil
}

type restartTestTPUCP struct {
	t *testing.T

	instance    *ctrl.TPUInstance
	instanceErr error

	createLRO *restartTestLRO
	deleteLRO *restartTestLRO

	calledDelete bool
	calledCreate bool
	createArg    string
	createErr    error
	deleteErr    error
}

func (r *restartTestTPUCP) Instance() (*ctrl.TPUInstance, error) {
	return r.instance, r.instanceErr
}

func (r *restartTestTPUCP) CreateInstance(version string) (ctrl.LongRunningOperation, error) {
	r.t.Helper()
	if r.calledCreate {
		r.t.Errorf("Already called CreateInstance")
	}
	r.calledCreate = true
	r.createArg = version
	return r.createLRO, r.createErr
}

func (r *restartTestTPUCP) DeleteInstance() (ctrl.LongRunningOperation, error) {
	r.t.Helper()
	if r.calledDelete {
		r.t.Errorf("Already called DeleteInstance")
	}
	r.calledDelete = true
	return r.deleteLRO, r.deleteErr
}

func TestRestartTPU(t *testing.T) {
	testcases := []struct {
		name string
		cfg  *config.Config
		node *tpu.Node
		want subcommands.ExitStatus
	}{{
		name: "normal",
		cfg: &config.Config{
			FlockName: "test-flock",
			Zone:      "us-central1-c",
			Project:   "test-project",
		},
		node: &tpu.Node{
			State:             "READY",
			TensorflowVersion: "1.8",
		},
		want: subcommands.ExitSuccess,
	}, {
		name: "missing-zone",
		cfg: &config.Config{
			FlockName: "test-flock",
			Zone:      "us-central1-c",
		},
		node: &tpu.Node{
			State:             "READY",
			TensorflowVersion: "1.8",
		},
		want: subcommands.ExitUsageError,
	}, {
		name: "no-tpu",
		cfg: &config.Config{
			FlockName: "test-flock",
			Zone:      "us-central1-c",
			Project:   "test-project",
		},
		node: nil,
		want: subcommands.ExitFailure,
	}, {
		name: "not-running",
		cfg: &config.Config{
			FlockName: "test-flock",
			Zone:      "us-central1-c",
			Project:   "test-project",
		},
		node: &tpu.Node{
			State:             "STOPPED",
			TensorflowVersion: "1.8",
		},
		want: subcommands.ExitFailure,
	}}

	for i, tt := range testcases {
		fmt.Printf("Starting testcase %d (%q)...\n", i, tt.name)
		var instance *ctrl.TPUInstance
		if tt.node != nil {
			instance = &ctrl.TPUInstance{tt.node}
		}
		createLRO := &restartTestLRO{}
		deleteLRO := &restartTestLRO{}
		cp := &restartTestTPUCP{
			t:         t,
			instance:  instance,
			createLRO: createLRO,
			deleteLRO: deleteLRO,
		}
		cmd := &restartCmd{tt.cfg, cp, true}
		resp := cmd.Execute(context.Background(), nil, nil)
		if resp != tt.want {
			t.Errorf("cmd.Execute(%q) = %v, want: %v", tt.name, resp, tt.want)
		}
		shouldCallCP := false
		if tt.want == subcommands.ExitSuccess {
			shouldCallCP = true
		}
		if cp.calledDelete != shouldCallCP {
			t.Errorf("cmd.Execute(%q).calledDelete = %v, want: %v", tt.name, cp.calledDelete, shouldCallCP)
		}
		if cp.calledCreate != shouldCallCP {
			t.Errorf("cmd.Execute(%q).calledCreate = %v, want: %v", tt.name, cp.calledCreate, shouldCallCP)
		}
		if shouldCallCP && cp.createArg != tt.node.TensorflowVersion {
			t.Errorf("cmd.Execute(%q).createArg = %v, want: %v", tt.name, cp.createArg, tt.node.TensorflowVersion)
		}
		if shouldCallCP && !createLRO.wait {
			t.Errorf("cmd.Execute(%q).createLRO.wait = %v, want: true", tt.name, createLRO.wait)
		}
		if shouldCallCP && !deleteLRO.wait {
			t.Errorf("cmd.Execute(%q).deleteLRO.wait = %v, want: true", tt.name, deleteLRO.wait)
		}
	}
}
