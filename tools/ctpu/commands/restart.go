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
	"fmt"
	"log"

	"flag"
	"github.com/google/subcommands"
	"github.com/tensorflow/tpu/tools/ctpu/config"
	"github.com/tensorflow/tpu/tools/ctpu/ctrl"
)

// RestartTPUCP abstracts the control plane interfaces required for the restart command.
type RestartTPUCP interface {
	// Instance retrieves the instance from the control plane (if available).
	Instance() (*ctrl.TPUInstance, error)
	// CreateInstance requests the creation of the instance.
	//
	// CreateInstance is implemented by TPUCP.CreateInstance and is expected to match.
	CreateInstance(ctx context.Context, version string, preemptible, reserved bool, hardwareType, network string) (ctrl.LongRunningOperation, error)
	// DeleteInstance requests the deletion of the instance.
	DeleteInstance() (ctrl.LongRunningOperation, error)
}

type restartCmd struct {
	cfg *config.Config
	tpu RestartTPUCP

	skipConfirmation bool
}

// RestartCommand creates the restart subcommand.
func RestartCommand(cfg *config.Config, tpu RestartTPUCP) subcommands.Command {
	return &restartCmd{
		cfg: cfg,
		tpu: tpu,
	}
}

func (restartCmd) Name() string     { return "restart" }
func (restartCmd) Synopsis() string { return "restarts your Cloud TPU" }
func (restartCmd) Usage() string    { return "ctpu restart\n" }

func (r *restartCmd) SetFlags(f *flag.FlagSet) {
	f.BoolVar(&r.skipConfirmation, "noconf", false, "Skip confirmation before restarting the Cloud TPU.")

	r.cfg.SetFlags(f)
}

func (r *restartCmd) Execute(ctx context.Context, flags *flag.FlagSet, args ...interface{}) subcommands.ExitStatus {
	err := r.cfg.Validate()
	if err != nil {
		log.Print(err)
		return subcommands.ExitUsageError
	}
	instance, err := r.tpu.Instance()
	if err != nil {
		log.Print(err)
		return subcommands.ExitFailure
	}
	if instance == nil {
		log.Printf("No Cloud TPU (name: %q, zone: %q) found. Aborting restart.", r.cfg.FlockName, r.cfg.Zone)
		return subcommands.ExitFailure
	}
	if !instance.IsRunning() {
		log.Printf("Your Cloud TPU (name: %q, zone: %q) does not appear to be running. (State: %q.) Aborting restart.", r.cfg.FlockName, r.cfg.Zone, instance.State)
		return subcommands.ExitFailure
	}
	version := instance.TensorflowVersion
	preemptible := instance.IsPreemptible()
	reserved := instance.IsReserved()
	tpuHardware := instance.AcceleratorType
	if version == "" {
		log.Printf("Your Cloud TPU (name: %q, zone: %q) does not appear to have a version. Aborting restart.", r.cfg.FlockName, r.cfg.Zone)
		return subcommands.ExitFailure
	}

	if !r.skipConfirmation {
		ok, err := askForConfirmation(fmt.Sprintf("About to delete and re-create your Cloud TPU (name: %q, zone: %q, version: %q). Ok?", r.cfg.FlockName, r.cfg.Zone, version))
		if err != nil {
			log.Fatalf("Restart confirmation error: %v", err)
		}
		if !ok {
			fmt.Printf("Exiting without making any changes.\n")
			return subcommands.ExitUsageError
		}
	}
	log.Printf("Deleting your Cloud TPU ahead of re-creating it...")
	op, err := r.tpu.DeleteInstance()
	if err != nil {
		log.Print(err)
		return subcommands.ExitFailure
	}
	err = op.LoopUntilComplete()
	if err != nil {
		log.Print(err)
		return subcommands.ExitFailure
	}
	log.Printf("Re-creating your Cloud TPU instance...")
	op, err = r.tpu.CreateInstance(ctx, version, preemptible, reserved, tpuHardware, instance.Network)
	if err != nil {
		log.Print(err)
		return subcommands.ExitFailure
	}
	err = op.LoopUntilComplete()
	if err != nil {
		log.Print(err)
		return subcommands.ExitFailure
	}
	log.Printf("Restart complete!")
	return subcommands.ExitSuccess
}
