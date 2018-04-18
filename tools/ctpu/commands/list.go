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
	"github.com/tensorflow/tpu/tools/ctpu/config"
	"github.com/tensorflow/tpu/tools/ctpu/ctrl"
)

// ListTPUInstancesCP lists the available TPU instances.
type ListTPUInstancesCP interface {
	// ListInstances lists the available TPU instances.
	ListInstances() ([]*ctrl.TPUInstance, error)
}

// ListGCEInstancesCP lists the available GCE instances.
type ListGCEInstancesCP interface {
	// ListInstances lists the available GCE instances.
	ListInstances() ([]*ctrl.GCEInstance, error)
}

type listCmd struct {
	cfg *config.Config
	tpu ListTPUInstancesCP
	gce ListGCEInstancesCP

	noHeader bool
	noColor  bool
}

// ListCommand creates the list command.
func ListCommand(cfg *config.Config, tpu ListTPUInstancesCP, gce ListGCEInstancesCP) subcommands.Command {
	return &listCmd{
		cfg: cfg,
		tpu: tpu,
		gce: gce,
	}
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

type listCmdAlias struct {
	listCmd
}

// ListCommandAlias creates an alias to the list command with a shorter name.
func ListCommandAlias(cfg *config.Config, tpu ListTPUInstancesCP, gce ListGCEInstancesCP) subcommands.Command {
	return &listCmdAlias{listCmd{cfg: cfg, tpu: tpu, gce: gce}}
}
func (listCmdAlias) Name() string     { return "ls" }
func (listCmdAlias) Synopsis() string { return "alias to ctpu list (lists all flocks)" }
func (listCmdAlias) Usage() string    { return "ctpu ls\n" }

func (c *listCmd) SetFlags(f *flag.FlagSet) {
	f.BoolVar(&c.noHeader, "no-header", false, "Do not print the header line.")
	f.BoolVar(&c.noColor, "no-color", false, "Disable color in the output.")
	c.cfg.SetFlags(f) // Allow users to specify cfg flags either before or after the subcommand name.
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
	err := c.cfg.Validate()
	if err != nil {
		log.Print(err)
		return subcommands.ExitFailure
	}

	if c.noColor {
		color.NoColor = true
	}

	flocks := make(map[string]*flock)

	vms, err := c.gce.ListInstances()

	if err != nil {
		log.Printf("Error listing GCE VM's: %v", err)
		return subcommands.ExitFailure
	}

	for _, vm := range vms {
		if vm.IsFlockVM() {
			flocks[vm.Name] = &flock{vm: vm}
		}
	}

	tpus, err := c.tpu.ListInstances()
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
		if flockName == c.cfg.FlockName {
			annotation = " (*)"
		}

		fmt.Printf("%d:\t%s%s\t%s\n", i, flockName, annotation, c.flockStatus(flock))
	}

	return subcommands.ExitSuccess
}
