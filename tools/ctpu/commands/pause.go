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
	"github.com/google/subcommands"
)

type pauseCmd struct {
	tpuCmd
}

// PauseCommand creates the pause command.
func PauseCommand() subcommands.Command {
	return &pauseCmd{}
}

func (pauseCmd) Name() string {
	return "pause"
}

func (pauseCmd) Synopsis() string {
	return "pause a Cloud TPU flock"
}

func (pauseCmd) Usage() string {
	return `ctpu pause [--dry-run] [--tpu-only] [--wait-for-async-ops]
`
}

func (c *pauseCmd) Execute(ctx context.Context, flags *flag.FlagSet, args ...interface{}) subcommands.ExitStatus {
	libs, err := parseArgs(args)
	if err != nil {
		log.Print(err)
		return subcommands.ExitFailure
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
			exitVM = cleanUpVM(libs.cfg, libs.gce, c.dryRun, "Stopping", libs.gce.StopInstance, c.waitForAsync)
		}
		wg.Done()
	}()

	wg.Wait()

	if exitTPU != 0 {
		return exitTPU
	}

	return exitVM
}
