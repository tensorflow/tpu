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
	"log"
	"sync"

	"context"
	"flag"
	"github.com/fatih/color"
	"github.com/google/subcommands"
	"github.com/tensorflow/tpu/tools/ctpu/config"
	"github.com/tensorflow/tpu/tools/ctpu/ctrl"
)

// DeleteGCECP abstracts the control plane interfaces required for the delete command.
type DeleteGCECP interface {
	// Instance retrieves the instance from the control plane (if available).
	Instance() (*ctrl.GCEInstance, error)
	// DeleteInstance requests the deletion of the instance.
	DeleteInstance() (ctrl.LongRunningOperation, error)
}

// DeleteTPUCP abstratcs the control plane interfaces required for the delete command.
type DeleteTPUCP interface {
	// Instance retrieves the instance from the control plane (if available).
	Instance() (*ctrl.TPUInstance, error)
	// DeleteInstance requests the deletion of the instance.
	DeleteInstance() (ctrl.LongRunningOperation, error)
}

type deleteCmd struct {
	cfg *config.Config
	gce DeleteGCECP
	tpu DeleteTPUCP

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

type deleteCmdAlias struct {
	deleteCmd
}

// DeleteCommandAlias creates an alias for the delete command with a shorter name.
func DeleteCommandAlias(cfg *config.Config, tpu DeleteTPUCP, gce DeleteGCECP) subcommands.Command {
	return &deleteCmdAlias{deleteCmd{cfg: cfg, gce: gce, tpu: tpu}}
}

func (deleteCmdAlias) Name() string     { return "rm" }
func (deleteCmdAlias) Synopsis() string { return "alias to ctpu delete (delete a Cloud TPU flock)" }
func (deleteCmdAlias) Usage() string    { return "ctpu rm\n" }

func (c *deleteCmd) Execute(ctx context.Context, flags *flag.FlagSet, args ...interface{}) subcommands.ExitStatus {
	err := c.cfg.Validate()
	if err != nil {
		log.Print(err)
		return subcommands.ExitFailure
	}

	if !c.tpuCmd.skipConfirmation {
		c.tpuCmd.printConfig(c.cfg)
		ok, err := askForConfirmation("About to permanently delete your resources. Ok?")
		if err != nil {
			log.Fatalf("Delete confirmation error: %v", err)
		}
		if !ok {
			color.Red("Exiting without making any changes.\n")
			return subcommands.ExitUsageError
		}
	}

	var tpuOp, gceOp ctrl.LongRunningOperation
	var tpuErr, gceErr error
	var wg sync.WaitGroup
	wg.Add(2)
	go func() {
		tpuOp, tpuErr = cleanUpTPU(c.cfg, c.tpu.Instance, c.tpu.DeleteInstance, c.dryRun)
		wg.Done()
	}()
	go func() {
		gceOp, gceErr = cleanUpVM(c.cfg, c.gce.Instance, c.gce.DeleteInstance, c.dryRun, "Deleting", false)
		wg.Done()
	}()
	wg.Wait()

	if tpuErr != nil {
		log.Print(tpuErr)
	}
	if gceErr != nil {
		log.Print(gceErr)
	}
	if tpuErr != nil || gceErr != nil {
		return subcommands.ExitFailure
	}

	err = waitForLongRunningOperations("delete", c.skipWaiting, gceOp, tpuOp)
	if err != nil {
		log.Print(err)
		return subcommands.ExitFailure
	}
	return subcommands.ExitSuccess
}
