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
)

type deleteCmd struct {
	skipConfirmation bool
	tpuCmd
}

// DeleteCommand creates the delete command.
func DeleteCommand() subcommands.Command {
	return &deleteCmd{}
}

func (deleteCmd) Name() string {
	return "delete"
}

func (d *deleteCmd) SetFlags(f *flag.FlagSet) {
	f.BoolVar(&d.skipConfirmation, "noconf", false, "Skip confirmation about deleting resources.")
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
	libs, err := parseArgs(args)
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
		exitTPU = cleanUpTPU(libs.cfg, libs.tpu, c.dryRun, c.waitForAsync)
		wg.Done()
	}()
	go func() {
		if !c.tpuOnly {
			exitVM = cleanUpVM(libs.cfg, libs.gce, c.dryRun, "Deleting", libs.gce.DeleteInstance, c.waitForAsync)
		}
		wg.Done()
	}()

	wg.Wait()

	if exitTPU != 0 {
		return exitTPU
	}
	return exitVM
}
