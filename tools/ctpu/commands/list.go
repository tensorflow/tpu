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
	"sort"

	"context"
	"flag"
	"github.com/fatih/color"
	"github.com/google/subcommands"
	"github.com/tensorflow/tpu/tools/ctpu/ctrl"
)

type listCmd struct {
	noHeader bool
	noColor  bool
}

// ListCommand creates the list command.
func ListCommand() subcommands.Command {
	return &listCmd{}
}

func (listCmd) Name() string {
	return "list"
}

func (listCmd) Synopsis() string {
	return "lists all Cloud TPU flocks"
}

func (listCmd) Usage() string {
	return `ctpu list
`
}

func (c *listCmd) SetFlags(f *flag.FlagSet) {
	f.BoolVar(&c.noHeader, "no-header", false, "Do not print the header line.")
	f.BoolVar(&c.noColor, "no-color", false, "Disable color in the output.")
}

type flock struct {
	vm  *ctrl.GCEInstance
	tpu *ctrl.TPUInstance
}

func (c *listCmd) flockStatus(flock *flock) string {
	if flock.vm == nil && flock.tpu == nil {
		return color.RedString("--")
	}
	if flock.vm != nil && flock.tpu == nil {
		return color.BlueString("paused")
	}
	if flock.vm != nil && flock.tpu != nil && flock.vm.IsRunning() && flock.tpu.IsRunning() {
		return color.GreenString("running")
	}
	return color.YellowString("unknown")
}

func (c *listCmd) Execute(ctx context.Context, flags *flag.FlagSet, args ...interface{}) subcommands.ExitStatus {
	libs, err := parseArgs(args)
	if err != nil {
		log.Printf("%v\n", err)
		return subcommands.ExitFailure
	}
	if c.noColor {
		color.NoColor = true
	}

	flocks := make(map[string]*flock)

	vms, err := libs.gce.ListInstances()

	if err != nil {
		log.Printf("Error listing GCE VM's: %v", err)
		return subcommands.ExitFailure
	}

	for _, vm := range vms {
		if vm.IsFlockVM() {
			flocks[vm.Name] = &flock{vm: vm}
		}
	}

	tpus, err := libs.tpu.ListInstances()
	if err != nil {
		log.Printf("Error listing Cloud TPUs: %v", err)
		return subcommands.ExitFailure
	}

	for _, tpu := range tpus {
		flock, ok := flocks[tpu.NodeName()]
		if ok {
			flock.tpu = tpu
		}
	}

	flockNames := make([]string, 0, len(flocks))
	for name, _ := range flocks {
		flockNames = append(flockNames, name)
	}
	sort.Strings(flockNames)

	if !c.noHeader {
		fmt.Printf("#\tFlock Name\tStatus\n")
	}
	for i, flockName := range flockNames {
		flock, ok := flocks[flockName]
		if !ok {
			log.Printf("Error retrieving flock name: %q\nFlocks: %v\n", flockName, flocks)
			return subcommands.ExitFailure
		}

		annotation := ""
		if flockName == libs.cfg.FlockName() {
			annotation = " (*)"
		}

		fmt.Printf("%d:\t%s%s\t%s\n", i, flockName, annotation, c.flockStatus(flock))
	}

	return subcommands.ExitSuccess
}
