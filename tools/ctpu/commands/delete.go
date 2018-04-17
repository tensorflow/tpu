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
	"log"
	"sync"

	"context"
	"flag"
	"github.com/google/subcommands"
	"github.com/tensorflow/tpu/tools/ctpu/config"
	"github.com/tensorflow/tpu/tools/ctpu/ctrl"
)

// DeleteGCECP abstracts the control plane interfaces required for the delete command.
type DeleteGCECP interface {
	// Instance retrieves the instance from the control plane (if available).
	Instance() (*ctrl.GCEInstance, error)
	// DeleteInstance requests the deletion of the instance.
	DeleteInstance(bool) error
}

// DeleteTPUCP abstratcs the control plane interfaces required for the delete command.
type DeleteTPUCP interface {
	// Instance retrieves the instance from the control plane (if available).
	Instance() (*ctrl.TPUInstance, error)
	// DeleteInstance requests the deletion of the instance.
	DeleteInstance(bool) error
}

type deleteCmd struct {
	cfg *config.Config
	gce DeleteGCECP
	tpu DeleteTPUCP

	skipConfirmation bool
	tpuCmd
}

// DeleteCommand creates the delete command.
func DeleteCommand(cfg *config.Config, tpu DeleteTPUCP, gce DeleteGCECP) subcommands.Command {
	return &deleteCmd{cfg: cfg, gce: gce, tpu: tpu}
}

func (deleteCmd) Name() string {
	return "delete"
}

func (d *deleteCmd) SetFlags(f *flag.FlagSet) {
	f.BoolVar(&d.skipConfirmation, "noconf", false, "Skip confirmation about deleting resources.")
	d.cfg.SetFlags(f) // Allow users to specify cfg flags either before or after the subcommand name.
	d.tpuCmd.SetFlags(f)
}

func (deleteCmd) Synopsis() string {
	return "delete a Cloud TPU flock"
}

func (deleteCmd) Usage() string {
	return `ctpu delete [--dry-run] [--tpu-only] [--wait-for-async-ops]
`
}

func (c *deleteCmd) Execute(ctx context.Context, flags *flag.FlagSet, args ...interface{}) subcommands.ExitStatus {
	err := c.cfg.Validate()
	if err != nil {
		log.Print(err)
		return subcommands.ExitFailure
	}

	if !c.skipConfirmation {
		ok, err := askForConfirmation("About to permanently delete your resources. Ok?")
		if err != nil {
			log.Fatalf("Delete confirmation error: %v", err)
		}
		if !ok {
			fmt.Printf("Exiting without making any changes.\n")
			return subcommands.ExitUsageError
		}
	}

	var exitTPU, exitVM subcommands.ExitStatus
	var wg sync.WaitGroup
	wg.Add(2)

	go func() {
		exitTPU = cleanUpTPU(c.cfg, c.tpu.Instance, c.tpu.DeleteInstance, c.dryRun, c.waitForAsync)
		wg.Done()
	}()
	go func() {
		if !c.tpuOnly {
			exitVM = cleanUpVM(c.cfg, c.gce.Instance, c.gce.DeleteInstance, c.dryRun, "Deleting", c.waitForAsync, false)
		}
		wg.Done()
	}()

	wg.Wait()

	if exitTPU != 0 {
		return exitTPU
	}
	return exitVM
}
