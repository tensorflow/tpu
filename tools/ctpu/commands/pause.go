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

// PauseGCECP abstracts the control plane interfaces required for the pause command.
type PauseGCECP interface {
	// Instance retrieves the instance from the control plane (if available).
	Instance() (*ctrl.GCEInstance, error)
	// StopInstance requests the halting of the instance.
	StopInstance() (ctrl.LongRunningOperation, error)
}

// PauseTPUCP abstracts the control plane interfaces required for the pause command.
type PauseTPUCP interface {
	// Instance retrieves the instance from the control plane (if available).
	Instance() (*ctrl.TPUInstance, error)
	// DeleteInstance requests the deletion of the instance.
	DeleteInstance() (ctrl.LongRunningOperation, error)
}

type pauseCmd struct {
	cfg *config.Config
	gce PauseGCECP
	tpu PauseTPUCP

	tpuCmd
}

// PauseCommand creates the pause command.
func PauseCommand(cfg *config.Config, tpu PauseTPUCP, gce PauseGCECP) subcommands.Command {
	return &pauseCmd{cfg: cfg, tpu: tpu, gce: gce}
}

func (pauseCmd) Name() string {
	return "pause"
}

func (pauseCmd) Synopsis() string {
	return "pause a Cloud TPU flock"
}

func (pauseCmd) Usage() string {
	return `ctpu pause [--dry-run] [--tpu-only] [--nowait] [--noconf]
`
}

type pauseCmdAlias struct {
	pauseCmd
}

// PauseCommandAlias creates an alias to the pause command with a shorter name.
func PauseCommandAlias(cfg *config.Config, tpu PauseTPUCP, gce PauseGCECP) subcommands.Command {
	return &pauseCmdAlias{pauseCmd{cfg: cfg, tpu: tpu, gce: gce}}
}

func (pauseCmdAlias) Name() string     { return "zz" }
func (pauseCmdAlias) Synopsis() string { return "alias to ctpu pause (pause a Cloud TPU flock)" }
func (pauseCmdAlias) Usage() string    { return "ctpu zz\n" }

func (c *pauseCmd) SetFlags(f *flag.FlagSet) {
	c.cfg.SetFlags(f) // Allow users to specify cfg flags either before or after the subcommand name.
	c.tpuCmd.SetFlags(f)
}

func (c *pauseCmd) Execute(ctx context.Context, flags *flag.FlagSet, args ...interface{}) subcommands.ExitStatus {
	err := c.cfg.Validate()
	if err != nil {
		log.Print(err)
		return subcommands.ExitFailure
	}

	if !c.tpuCmd.skipConfirmation {
		c.tpuCmd.printConfig(c.cfg)
		ok, err := askForConfirmation("About to shut down your resources. OK?")
		if err != nil {
			log.Fatalf("Pause confirmation error: %v", err)
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
		if !c.tpuOnly {
			gceOp, gceErr = cleanUpVM(c.cfg, c.gce.Instance, c.gce.StopInstance, c.dryRun, "Stopping", true)
		}
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

	err = waitForLongRunningOperations("pause", c.skipWaiting, gceOp, tpuOp)
	if err != nil {
		log.Print(err)
		return subcommands.ExitFailure
	}
	return subcommands.ExitSuccess
}
