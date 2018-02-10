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
	"strings"

	"context"
	"flag"
	"github.com/fatih/color"
	"github.com/google/subcommands"
	"github.com/tensorflow/tpu/tools/ctpu/ctrl"
)

type sprintfFunc func(format string, a ...interface{}) string

type statusCmd struct {
	details bool
	noColor bool
}

// StatusCommand creates the status command.
func StatusCommand() subcommands.Command {
	return &statusCmd{}
}

func (statusCmd) Name() string {
	return "status"
}

func (s *statusCmd) SetFlags(f *flag.FlagSet) {
	f.BoolVar(&s.details, "details", false,
		"Prints out more details about the state of the GCE VM and Cloud TPU.")
	f.BoolVar(&s.noColor, "no-color", false, "Disable color in the output.")
}

func (statusCmd) Synopsis() string {
	return "queries the control planes for the current GCE & TPU status."
}

func (statusCmd) Usage() string {
	return `ctpu status [--no-color]
`
}

func (s *statusCmd) runnableStatus(exists, isRunning bool, status string) string {
	if !exists {
		return color.YellowString("--")
	}
	if isRunning {
		return color.GreenString("RUNNING")
	}
	return color.RedString(status)
}

func (s *statusCmd) vmStatus(vm *ctrl.GCEInstance) string {
	var status string
	if vm != nil {
		status = vm.Status
	}
	exists := vm != nil
	isRunning := vm != nil && vm.IsRunning()
	return s.runnableStatus(exists, isRunning, status)
}

func (s *statusCmd) flockStatus(vm *ctrl.GCEInstance, tpu *ctrl.TPUInstance) string {
	if vm == nil && tpu == nil {
		return color.BlueString("No instances currently exist.")
	}
	if vm != nil && vm.IsRunning() && tpu != nil && tpu.IsRunning() {
		return color.GreenString("Your cluster is running!")
	}
	if vm != nil && !vm.IsRunning() && tpu == nil {
		return color.YellowString("Your cluster is paused.")
	}
	return color.RedString("Your cluster is in an unhealthy state.")
}

func (s *statusCmd) tpuStatus(tpu *ctrl.TPUInstance) string {
	var status string
	if tpu != nil {
		status = tpu.State
	}
	exists := tpu != nil
	isRunning := tpu != nil && tpu.IsRunning()
	return s.runnableStatus(exists, isRunning, status)
}

func (s *statusCmd) Execute(ctx context.Context, flags *flag.FlagSet, args ...interface{}) subcommands.ExitStatus {
	libs, err := parseArgs(args)
	if err != nil {
		log.Print(err)
		return subcommands.ExitFailure
	}
	if s.noColor {
		color.NoColor = true
	}

	vm, err := libs.gce.Instance()
	if err != nil {
		log.Print(err)
		return subcommands.ExitFailure
	}

	tpu, err := libs.tpu.Instance()
	if err != nil {
		log.Print(err)
		return subcommands.ExitFailure
	}

	fmt.Printf(`%s
	GCE VM:     %s
	Cloud TPU:  %s
`, s.flockStatus(vm, tpu), s.vmStatus(vm), s.tpuStatus(tpu))

	vmIP, vmCreated, machineType := "--", "--", "--"
	if vm != nil {
		if len(vm.NetworkInterfaces) > 0 {
			vmIP = vm.NetworkInterfaces[0].NetworkIP
		}
		// TODO(saeta): Print delta between now and creation time.
		vmCreated = vm.CreationTimestamp

		machineTypeParts := strings.Split(vm.MachineType, "/")
		machineType = machineTypeParts[len(machineTypeParts)-1]
	}

	tpuType, tpuIP, tpuVer, tpuSA, tpuCreated, tpuState, tpuHealth := "--", "--", "--", "--", "--", "--", "--"
	if tpu != nil {
		tpuType = tpu.AcceleratorType
		if len(tpu.NetworkEndpoints) > 0 {
			tpuIP = tpu.NetworkEndpoints[0].IpAddress
		}
		tpuVer = tpu.TensorflowVersion
		tpuSA = tpu.ServiceAccount
		// TODO(saeta): Print delta between now and creation time.
		tpuCreated = tpu.CreateTime
		tpuState = tpu.State
		tpuHealth = tpu.Health
	}

	if s.details {
		fmt.Printf(`
GCE IP Address:        %s
GCE Created:           %s
GCE Machine Type:      %s
TPU Accelerator Type:  %s
TPU IP Address:        %s
TPU TF Version:        %s
TPU Service Acct:      %s
TPU Created:           %s
TPU State:             %s
TPU Health:            %s
`, vmIP, vmCreated, machineType, tpuType, tpuIP, tpuVer, tpuSA, tpuCreated, tpuState, tpuHealth)
	}
	return subcommands.ExitSuccess
}
