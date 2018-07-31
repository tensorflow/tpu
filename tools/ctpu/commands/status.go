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
	"sync"
	"time"

	"context"
	"flag"
	"github.com/fatih/color"
	"github.com/google/subcommands"
	"github.com/tensorflow/tpu/tools/ctpu/config"
	"github.com/tensorflow/tpu/tools/ctpu/ctrl"
)

// StatusTPUCP encapsulates the control plane interfaces required to execute the Status command.
type StatusTPUCP interface {
	// Instance retrieves the TPU instance (if available).
	Instance() (*ctrl.TPUInstance, error)
}

// StatusGCECP encapsulates the control plane interfaces required to execute the Status command.
type StatusGCECP interface {
	// Instance retrieves the Compute Engine instance (if available).
	Instance() (*ctrl.GCEInstance, error)
}

type statusCmd struct {
	cfg *config.Config
	tpu StatusTPUCP
	gce StatusGCECP

	details bool
	noColor bool
}

// StatusCommand creates the status command.
func StatusCommand(cfg *config.Config, tpu StatusTPUCP, gce StatusGCECP) subcommands.Command {
	return &statusCmd{
		cfg: cfg,
		tpu: tpu,
		gce: gce,
	}
}

func (statusCmd) Name() string {
	return "status"
}

func (s *statusCmd) SetFlags(f *flag.FlagSet) {
	s.cfg.SetFlags(f) // Allow users to specify cfg flags either before or after the subcommand name.
	f.BoolVar(&s.details, "details", false,
		"Prints out more details about the state of the Compute Engine VM and Cloud TPU.")
	f.BoolVar(&s.noColor, "no-color", false, "Disable color in the output.")
}

func (statusCmd) Synopsis() string {
	return "queries the control planes for the current Compute Engine & TPU status."
}

func (statusCmd) Usage() string {
	return `ctpu status [--no-color]
`
}

type statusCmdAlias struct {
	statusCmd
}

// StatusCommandAlias creates an alias for the status command with a shorter name.
func StatusCommandAlias(cfg *config.Config, tpu StatusTPUCP, gce StatusGCECP) subcommands.Command {
	return &statusCmdAlias{statusCmd{cfg: cfg, tpu: tpu, gce: gce}}
}

func (statusCmdAlias) Name() string { return "st" }
func (statusCmdAlias) Synopsis() string {
	return "alias to ctpu status (retrieves info on current instances)"
}
func (statusCmdAlias) Usage() string { return "ctpu st\n" }

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

func (s *statusCmd) timeDelta(t time.Time) string {
	delta := time.Since(t).Round(time.Minute)
	if delta < 0 {
		return "--"
	}
	if delta.Minutes() < 1 {
		return "< 1 minute"
	}
	minutes := (delta / time.Minute) % 60
	hours := delta / time.Hour
	days := delta / (time.Hour * 24)

	if days > 3 {
		return fmt.Sprintf("%dd %dh", days, hours%24)
	}
	if hours == 0 {
		return fmt.Sprintf("%dm", minutes)
	}
	return fmt.Sprintf("%dh %dm", hours, minutes)
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
	err := s.cfg.Validate()
	if err != nil {
		log.Print(err)
		return subcommands.ExitFailure
	}

	if s.noColor {
		color.NoColor = true
	}

	var vm *ctrl.GCEInstance
	var tpu *ctrl.TPUInstance
	var exitTPU, exitVM subcommands.ExitStatus
	var wg sync.WaitGroup
	wg.Add(2)

	go func() {
		var err error
		vm, err = s.gce.Instance()
		if err != nil {
			log.Print(err)
			exitVM = subcommands.ExitFailure
		}
		wg.Done()
	}()

	go func() {
		var err error
		tpu, err = s.tpu.Instance()
		if err != nil {
			log.Print(err)
			exitTPU = subcommands.ExitFailure
		}
		wg.Done()
	}()

	wg.Wait()
	if exitTPU != subcommands.ExitSuccess {
		return exitTPU
	}
	if exitVM != subcommands.ExitSuccess {
		return exitVM
	}

	fmt.Printf(`%s
	Compute Engine VM:  %s
	Cloud TPU:          %s
`, s.flockStatus(vm, tpu), s.vmStatus(vm), s.tpuStatus(tpu))

	vmIP, vmCreated, vmCreateDelta, machineType := "--", "--", "--", "--"
	if vm != nil {
		if len(vm.NetworkInterfaces) > 0 {
			vmIP = vm.NetworkInterfaces[0].NetworkIP
		}
		vmCreated = vm.CreationTimestamp
		if createTime, err := time.Parse(time.RFC3339, vmCreated); err == nil {
			vmCreateDelta = s.timeDelta(createTime)
		}

		machineTypeParts := strings.Split(vm.MachineType, "/")
		machineType = machineTypeParts[len(machineTypeParts)-1]
	}

	tpuType, tpuIP, tpuVer, tpuSA, tpuCreated, tpuCreateDelta, tpuState, tpuHealth, tpuPreemptible := "--", "--", "--", "--", "--", "--", "--", "--", "--"
	if tpu != nil {
		tpuType = tpu.AcceleratorType
		if len(tpu.NetworkEndpoints) > 0 {
			tpuIP = tpu.NetworkEndpoints[0].IpAddress
		}
		tpuVer = tpu.TensorflowVersion
		tpuSA = tpu.ServiceAccount
		tpuCreated = tpu.CreateTime
		if createTime, err := time.Parse(time.RFC3339Nano, tpuCreated); err == nil {
			tpuCreateDelta = s.timeDelta(createTime)
		}
		tpuState = tpu.State
		tpuHealth = tpu.Health
		tpuPreemptible = fmt.Sprintf("%v", tpu.IsPreemptible())
	}

	if s.details {
		fmt.Printf(`
Compute Engine IP Address:    %s
Compute Engine Created:       %s ago (@: %s)
Compute Engine Machine Type:  %s
TPU Accelerator Type:         %s
TPU IP Address:               %s
TPU TF Version:               %s
TPU Service Acct:             %s
TPU Created:                  %s ago (@: %s)
TPU State:                    %s
TPU Health:                   %s
TPU Preemptible:              %s
`, vmIP, vmCreateDelta, vmCreated, machineType, tpuType, tpuIP, tpuVer, tpuSA, tpuCreateDelta, tpuCreated, tpuState, tpuHealth, tpuPreemptible)
	}
	return subcommands.ExitSuccess
}
