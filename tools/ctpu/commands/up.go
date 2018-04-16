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
	"github.com/google/subcommands"
	"github.com/tensorflow/tpu/tools/ctpu/config"
	"github.com/tensorflow/tpu/tools/ctpu/ctrl"
	"golang.org/x/sync/errgroup"
	"google.golang.org/api/tpu/v1alpha1"
)

// UpTPUCP abstracts the control plane interface required for the up command.
type UpTPUCP interface {
	// Instance retrieves the instance from the control plane (if available).
	Instance() (*ctrl.TPUInstance, error)
	// CreateInstance requests the creation of the instance.
	CreateInstance(version string) error
	// ListVersions retrieves the list of available TensorFlow versions.
	ListVersions() ([]*tpu.TensorFlowVersion, error)
}

// UpGCECP abstracts the control plane interface requred for the up command.
type UpGCECP interface {
	// Instance retrieves the instance from the control plane (if available).
	Instance() (*ctrl.GCEInstance, error)
	// CreateInstance requests the creation of the instance.
	CreateInstance(*ctrl.GCECreateRequest) error
	// StartInstance requests the starting of the instance.
	StartInstance() error
}

// UpResourceManagementCP abstracts the control plane interface required for the up command.
type UpResourceManagementCP interface {
	// AddTPUUserAgent authorizes the TPU user agent, if required.
	AddTPUUserAgent(tpuUserAgent string) error
}

// UpGcloudCLI abstracts the interaction with the gcloud command line tools for the up command.
type UpGcloudCLI interface {
	// IsGcloudInstalled determines if the gcloud command line tool is installed and available on the path.
	IsGcloudInstalled() bool
	// SSHToInstance takes over the process and turns it into an ssh connection to the GCE instance of the flock.
	SSHToInstance(forwardPorts, forwardAgent bool, tpuInstance *ctrl.TPUInstance) error
	// PrintInstallInstructions prints instructions to the console for how to install gcloud.
	PrintInstallInstructions()
}

type upCmd struct {
	cfg *config.Config
	tpu UpTPUCP
	gce UpGCECP
	rmg UpResourceManagementCP
	cli UpGcloudCLI

	dryRun       bool
	vmOnly       bool
	forwardPorts bool
	forwardAgent bool
	tfVersion    string
	gceImage     string
	diskSizeGb   int64
	machineType  string
}

// UpCommand creates the up command.
func UpCommand(cfg *config.Config, tpu UpTPUCP, gce UpGCECP, rmg UpResourceManagementCP, cli UpGcloudCLI) subcommands.Command {
	return &upCmd{
		cfg: cfg,
		tpu: tpu,
		gce: gce,
		rmg: rmg,
		cli: cli,
	}
}

func (upCmd) Name() string {
	return "up"
}

func (c *upCmd) SetFlags(f *flag.FlagSet) {
	f.BoolVar(&c.dryRun, "dry-run", false,
		"Do not make changes; print only what would have happened.")
	f.BoolVar(&c.vmOnly, "vm-only", false,
		"Do not allocate a TPU, only allocate a VM (useful if you're not ready to run on a TPU yet).")
	f.BoolVar(&c.forwardPorts, "forward-ports", true,
		"Automatically forward useful ports from the GCE VM to your local machine. The ports forwarded are: 6006 (tensorboard), 8888 (jupyter notebooks), 8470 (TPU port), 8466 (TPU profiler port).")
	f.BoolVar(&c.forwardAgent, "forward-agent", true,
		"Enable ssh agent forwarding when sshing into the GCE VM. (SSH Agent Forwarding enables access to shared repositories (e.g. github) without having to place private keys on the GCE VM.)")
	f.StringVar(&c.tfVersion, "tf-version", "",
		"Set the version of TensorFlow to use when creating the GCE VM and the Cloud TPU. (It defaults to auto-selecting the latest stable release.)")
	f.StringVar(&c.gceImage, "gce-image", "",
		"Override the automatically chosen GCE Image. Use this flag when you're using your own custom images instead of the provided ones with TensorFlow pre-installed.")
	f.Int64Var(&c.diskSizeGb, "disk-size-gb", 250, "Configures the root volume size of your GCE VM.")
	f.StringVar(&c.machineType, "machine-type", "n1-standard-2", "Configures the size of your GCE VM.")
}

func (upCmd) Synopsis() string {
	return "bring up a Cloud TPU flock (VM + TPU)."
}

func (upCmd) Usage() string {
	return `ctpu up [--dry-run]
`
}

func (c *upCmd) gceImageFamily() (string, error) {
	if c.tfVersion == "" {
		return "", fmt.Errorf("invalid tensorflow version %q", c.tfVersion)
	}
	if c.tfVersion == "nightly" {
		return "tf-nightly", nil
	}
	parsed, err := parseVersion(c.tfVersion)
	if err != nil {
		return "", err
	}
	if parsed.Modifier != "" {
		return "", fmt.Errorf("invalid tensorflow version %q (non-empty modifier)", c.tfVersion)
	}
	return fmt.Sprintf("tf-%d-%d", parsed.Major, parsed.Minor), nil
}

func (c *upCmd) upVM() error {
	// Create the GCE VM
	vm, err := c.gce.Instance()
	if err != nil {
		log.Print(err)
		return err
	}
	if vm == nil {
		log.Printf("Creating GCE VM %s (this may take a minute)...\n", c.cfg.FlockName)
		imageFamily, err := c.gceImageFamily()
		if err != nil {
			log.Print(err)
			return err
		}
		if !c.dryRun {
			req := &ctrl.GCECreateRequest{
				TensorFlowVersion: c.tfVersion,
				ImageFamily:       imageFamily,
				ImageName:         c.gceImage,
				MachineType:       c.machineType,
				DiskSizeGb:        c.diskSizeGb,
			}
			err = c.gce.CreateInstance(req)
			if err != nil {
				log.Print(err)
				return err
			}
		}
		log.Printf("Created GCE VM %s!\n", c.cfg.FlockName)
	} else {
		if !strings.HasSuffix(vm.Instance.MachineType, c.machineType) {
			log.Printf("Warning: GCE VM machine type is not %q: actual: %q", c.machineType, vm.Instance.MachineType)
		}

		if !vm.IsRunning() {
			log.Printf("Starting GCE VM %s...\n", c.cfg.FlockName)
			if !c.dryRun {
				err = c.gce.StartInstance()
			}
			if err != nil {
				log.Print(err)
				return err
			}
			log.Printf("Started GCE VM %s!\n", c.cfg.FlockName)
		} else {
			log.Printf("VM already running.")
		}
	}
	return nil
}

func (c *upCmd) upTPU() (*ctrl.TPUInstance, error) {
	tpu, err := c.tpu.Instance()
	if err != nil {
		log.Printf("%v", err)
		return nil, err
	}
	if tpu == nil {
		log.Printf("Creating TPU %s (this may take a few minutes)...\n", c.cfg.FlockName)
		if !c.dryRun {
			err = c.tpu.CreateInstance(c.tfVersion)
			if err != nil {
				log.Printf("%v", err)
				return nil, err
			}
		}
		log.Printf("Created TPU %s!\n", c.cfg.FlockName)

		// Refresh the instance to get the service account the TPU runs as.
		tpu, err = c.tpu.Instance()
		if err != nil {
			log.Printf("error refreshing TPU instance after creation - %#v", err)
			return nil, err
		}
		if tpu == nil {
			err := fmt.Errorf("error creating TPU, please try again later")
			log.Printf("%v", err)
			return nil, err
		}
		err = c.rmg.AddTPUUserAgent(tpu.ServiceAccount)
		if err != nil {
			log.Printf("error adding the TPU's service account to the project's access control lists: %#v", err)
			return nil, err
		}
	} else if !tpu.IsRunning() {
		err := fmt.Errorf("the TPU exists, but is not running... aborting")
		log.Printf("%v", err)
		return nil, err
	} else {
		log.Printf("TPU already running.")
	}

	return tpu, nil
}

// ConfigureTFVersion ensures the selected version is appropriate.
func (c *upCmd) ConfigureTFVersion() error {
	rawVersions, err := c.tpu.ListVersions()
	if err != nil {
		return err
	}

	if c.tfVersion == "" {
		parsedVersions := make([]parsedVersion, 0, len(rawVersions))
		for _, version := range rawVersions {
			parsed, err := parseVersion(version.Version)
			if err != nil {
				return err
			}
			parsedVersions = append(parsedVersions, parsed)
		}
		sortedVersions := sortedParsedVersions(parsedVersions)
		c.tfVersion, err = sortedVersions.LatestStableRelease()
		return err
	}

	for _, version := range rawVersions {
		if c.tfVersion == version.Version {
			return nil
		}
	}

	return fmt.Errorf("%q TensorFlow version not available; to see available versions, execute ctpu tf-versions", c.tfVersion)
}

func (c *upCmd) Execute(ctx context.Context, flags *flag.FlagSet, args ...interface{}) subcommands.ExitStatus {
	err := c.cfg.Validate()
	if err != nil {
		log.Print(err)
		return subcommands.ExitFailure
	}

	if !c.cli.IsGcloudInstalled() {
		c.cli.PrintInstallInstructions()
		return subcommands.ExitFailure
	}

	// TF Version check.
	err = c.ConfigureTFVersion()
	if err != nil {
		log.Print(err)
		return subcommands.ExitFailure
	}

	var tpu *ctrl.TPUInstance
	g, ctx := errgroup.WithContext(ctx)

	// Create the GCE VM
	g.Go(func() error {
		return c.upVM()
	})

	// Create the Cloud TPU
	g.Go(func() (err error) {
		tpu, err = c.upTPU()
		return err
	})

	if err := g.Wait(); err != nil {
		fmt.Printf("%v\n", err)
		return subcommands.ExitFailure
	}

	if c.forwardPorts {
		fmt.Println("About to ssh (with port forwarding enabled -- see docs details)...")
	} else {
		fmt.Println("About to ssh...")
	}
	if err := c.cli.SSHToInstance(c.forwardPorts, c.forwardAgent, tpu); err != nil {
		log.Printf("Could not ssh to instance: %v\n", err)
		return subcommands.ExitFailure
	}
	return subcommands.ExitSuccess
}
