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
	"regexp"
	"strings"

	"context"
	"flag"
	"github.com/fatih/color"
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
	// OptionallyRetrieveInstance retrieves the instance, but can optionally not enable the TPU API.
	OptionallyRetrieveInstance(bool) (*ctrl.TPUInstance, bool, error)
	// CreateInstance requests the creation of the instance.
	CreateInstance(ctx context.Context, version string, preemptible bool, hardwareType string) (ctrl.LongRunningOperation, error)
	// ListVersions retrieves the list of available TensorFlow versions.
	ListVersions() ([]*tpu.TensorFlowVersion, error)
}

// UpGCECP abstracts the control plane interface requred for the up command.
type UpGCECP interface {
	// Instance retrieves the instance from the control plane (if available).
	Instance() (*ctrl.GCEInstance, error)
	// OptionallyRetrieveInstance retrieves the instance, but can optionally not enable the TPU API.
	OptionallyRetrieveInstance(bool) (*ctrl.GCEInstance, bool, error)
	// CreateInstance requests the creation of the instance.
	CreateInstance(*ctrl.GCECreateRequest) (ctrl.LongRunningOperation, error)
	// StartInstance requests the starting of the instance.
	StartInstance() (ctrl.LongRunningOperation, error)
}

// UpResourceManagementCP abstracts the control plane interface required for the up command.
type UpResourceManagementCP interface {
	// AddTPUUserAgent authorizes the TPU user agent, if required.
	AddTPUUserAgent(tpuUserAgent string) error

	// IsProjectInGoogleOrg determines if the project is part of the Google organziation.
	//
	// This is used to provide more helpful error messages
	IsProjectInGoogleOrg() (bool, error)
}

// UpGcloudCLI abstracts the interaction with the gcloud command line tools for the up command.
type UpGcloudCLI interface {
	// IsGcloudInstalled determines if the gcloud command line tool is installed and available on the path.
	IsGcloudInstalled() bool
	// SSHToInstance takes over the process and turns it into an ssh connection to the Compute Engine instance of the flock.
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

	// Common parameters
	skipConfirmation bool
	printWelcome     bool
	dryRun           bool
	vmOnly           bool
	tpuOnly	         bool
	forwardPorts     bool
	forwardAgent     bool
	tfVersion        string
	requirePerms     bool

	// VM parameters
	gceImage      string
	dlImage       bool // Use the Deep Learning VM images instead of the TPU specific ones.
	diskSizeGb    int64
	machineType   string
	preemptibleVM bool

	// TPU parameters
	preemptibleTPU bool
	tpuHardware    string
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
	c.cfg.SetFlags(f) // Allow users to specify cfg flags either before or after the subcommand name.
	f.BoolVar(&c.skipConfirmation, "noconf", false, "Skip confirmation when running the up subcommand.")
	f.BoolVar(&c.printWelcome, "print-welcome", false, "Always print the welcome message.")
	f.BoolVar(&c.dryRun, "dry-run", false,
		"Do not make changes; print only what would have happened.")
	f.BoolVar(&c.vmOnly, "vm-only", false,
		"Do not allocate a TPU, only allocate a VM (useful if you're not ready to run on a TPU yet).")
	f.BoolVar(&c.tpuOnly, "tpu-only", false,
		"Do not allocate a VM, only allocate a TPU (useful if you already have a VM ready).")
	f.BoolVar(&c.forwardPorts, "forward-ports", true,
		"Automatically forward useful ports from the Compute Engine VM to your local machine. The ports forwarded are: 6006 (tensorboard), 8888 (jupyter notebooks), 8470 (TPU port), 8466 (TPU profiler port).")
	f.BoolVar(&c.forwardAgent, "forward-agent", true,
		"Enable ssh agent forwarding when sshing into the Compute Engine VM. (SSH Agent Forwarding enables access to shared repositories (e.g. github) without having to place private keys on the Compute Engine VM.)")
	f.StringVar(&c.tfVersion, "tf-version", "",
		"Set the version of TensorFlow to use when creating the Compute Engine VM and the Cloud TPU. (It defaults to auto-selecting the latest stable release.)")
	// Note: because some users won't be OWNERS of their projects, and because bucket-level permissions can be used, default is set to not require permissions.
	f.BoolVar(&c.requirePerms, "require-permissions", false, "Stop the TPU setup if modification of Cloud IAM permissions fails. By default, ctpu prints a warning message then continues with the setup.")

	f.StringVar(&c.gceImage, "gce-image", "",
		"Override the automatically chosen Compute Engine Image. Use this flag when you're using your own custom images instead of the provided ones with TensorFlow pre-installed.")
	f.BoolVar(&c.dlImage, "use-dl-images", false, "Use Deep Learning VM Images (see docs: https://cloud.google.com/deep-learning-vm/) instead of TPU-specific machine images. Defaults to TPU-specific images.")
	f.Int64Var(&c.diskSizeGb, "disk-size-gb", 250, "Configures the root volume size of your Compute Engine VM.")
	f.StringVar(&c.machineType, "machine-type", "n1-standard-2", "Configures the size of your Compute Engine VM.")
	f.BoolVar(&c.preemptibleVM, "preemptible-vm", false, "Create a preemptible Compute Engine VM, instead of a normal (non-preemptible) VM. A preemptible VM costs less per hour, but the Compute Engine service can terminate the instance at any time.")

	f.BoolVar(&c.preemptibleTPU, "preemptible", false, "Create a preemptible Cloud TPU, instead of a normal (non-preemptible) Cloud TPU. A preemptible Cloud TPU costs less per hour, but the Cloud TPU service can stop/terminate the node at any time.")
	f.StringVar(&c.tpuHardware, "tpu-size", "v2-8", "Configure the size and generation of the Cloud TPU.")
}

func (upCmd) Synopsis() string {
	return "bring up a Cloud TPU flock (VM + TPU)."
}

func (upCmd) Usage() string {
	return `ctpu up [--noconf] [--dry-run]
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
		return "", fmt.Errorf("invalid tensorflow version %q (non-empty modifier); please set the --gce-image flag", c.tfVersion)
	}
	if c.dlImage {
		return fmt.Sprintf("tf-%d-%d-gpu", parsed.Major, parsed.Minor), nil
	}
	return fmt.Sprintf("tf-%d-%d", parsed.Major, parsed.Minor), nil
}

func (upCmd) printFirstTimeMessages() {
	c := color.New(color.Bold, color.Underline)
	fmt.Printf("\n         ")
	c.Println("Welcome to ctpu!")
	fmt.Printf(`
After confirming the configuration looks correct, ctpu will enable the
necessary service APIs, start a Cloud TPU with the latest released TensorFlow
version, and start a Compute Engine VM with a compatible version of TensorFlow
pre-installed. When everything is ready, ctpu will automatically open an ssh
connection to your new VM and port-forward commonly used ports. For more
details, see the documentation at:
      https://github.com/tensorflow/tpu/tools/ctpu

`)
}

var fullGCEImageRegex = regexp.MustCompile("https://www.googleapis.com/compute/v1/projects/[^/]+/global/images/[^/]+")

func (c *upCmd) canonicalizeGCEImage() error {
	if c.gceImage == "" {
		return nil
	}

	if fullGCEImageRegex.MatchString(c.gceImage) {
		return nil
	}

	splits := strings.Split(c.gceImage, "/")
	if len(splits) > 2 {
		return fmt.Errorf("invalid GCE Image specified (%q); must be either (1) a fully specified URL, (2) \"image_name\", or (3) \"project_Name/image_name\"", c.gceImage)
	}

	var imageName, projectName string
	if len(splits) == 1 {
		imageName = splits[0]
		projectName = c.cfg.Project
	}
	if len(splits) == 2 {
		imageName = splits[1]
		projectName = splits[0]
	}
	c.gceImage = fmt.Sprintf("https://www.googleapis.com/compute/v1/projects/%s/global/images/%s", projectName, imageName)

	return nil
}

func (c *upCmd) upVM() error {
	if c.tpuOnly {
		return nil
	}
	// Create the Compute Engine VM
	vm, err := c.gce.Instance()
	if err != nil {
		log.Print(err)
		return err
	}
	if vm == nil {
		log.Printf("Creating Compute Engine VM %s (this may take a minute)...\n", c.cfg.FlockName)
		imageFamily, err := c.gceImageFamily()
		if err != nil && c.gceImage == "" {
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
				Preemptible:       c.preemptibleVM,
			}
			op, err := c.gce.CreateInstance(req)
			if err != nil {
				log.Print(err)
				return err
			}
			err = op.LoopUntilComplete()
			if err != nil {
				log.Print(err)
				return err
			}
		}
		log.Printf("Created Compute Engine VM %s!\n", c.cfg.FlockName)
	} else {
		if !strings.HasSuffix(vm.Instance.MachineType, c.machineType) {
			log.Printf("Warning: Compute Engine VM machine type is not %q: actual: %q", c.machineType, vm.Instance.MachineType)
		}

		if !vm.IsRunning() {
			log.Printf("Starting Compute Engine VM %s...\n", c.cfg.FlockName)
			if !c.dryRun {
				op, err := c.gce.StartInstance()
				if err != nil {
					log.Print(err)
					return err
				}
				err = op.LoopUntilComplete()
				if err != nil {
					log.Print(err)
					return err
				}
			}
			log.Printf("Started Compute Engine VM %s!\n", c.cfg.FlockName)
		} else {
			log.Printf("VM already running.")
		}
	}
	return nil
}

func (c *upCmd) upTPU(ctx context.Context) (*ctrl.TPUInstance, error) {
	if c.vmOnly {
		return nil, nil
	}
	tpu, err := c.tpu.Instance()
	if err != nil {
		log.Printf("%v", err)
		return nil, err
	}
	if tpu == nil {
		log.Printf("Creating TPU %s (this may take a few minutes)...\n", c.cfg.FlockName)
		if !c.dryRun {
			op, err := c.tpu.CreateInstance(ctx, c.tfVersion, c.preemptibleTPU, c.tpuHardware)
			if err != nil {
				log.Print(err)
				return nil, err
			}
			err = op.LoopUntilComplete()
			if err != nil {
				log.Print(err)
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
			if c.requirePerms {
				// Error out.
				log.Printf("Error adding the TPU's service account to the project's access control lists: %#v", err)
				return nil, err
			}
			log.Print("Warning: ctpu encountered an error when adding the TPU's service account to your project's access control lists. Some integrations (for example: Cloud Storage) may fail until you (or your GCP project's owner) adds appropriate permissions (see: https://cloud.google.com/tpu/docs/storage-buckets#storage_access). Pass --require-permissions to turn this warning into an error and get a more detailed error message.")
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

func (c *upCmd) launchInstances(ctx context.Context) (*ctrl.TPUInstance, error) {
	var tpu *ctrl.TPUInstance
	g, ctx := errgroup.WithContext(ctx)

	// Create the Compute Engine VM
	g.Go(func() error {
		return c.upVM()
	})

	// Create the Cloud TPU
	g.Go(func() (err error) {
		tpu, err = c.upTPU(ctx)
		return err
	})

	if err := g.Wait(); err != nil {
		return nil, err
	}
	return tpu, nil
}

func (c *upCmd) confirmExecution(tpuAPIEnabled bool) (bool, error) {
	if !tpuAPIEnabled || c.printWelcome {
		c.printFirstTimeMessages()
	}
	tfVersion := c.tfVersion
	if !tpuAPIEnabled && tfVersion == "" {
		tfVersion = "<set after TPU API enabled>"
	}
	fmt.Printf(`ctpu will use the following configuration:

  Name:                 %s
  Zone:                 %s
  GCP Project:          %s
  TensorFlow Version:   %s
  VM:
      Machine Type:     %s
      Disk Size:        %d GB
      Preemptible:      %v
`, c.cfg.FlockName, c.cfg.Zone, c.cfg.Project, tfVersion, c.machineType, c.diskSizeGb,
		c.preemptibleVM)
	if !c.vmOnly {
		fmt.Printf(`  Cloud TPU:
      Size:             %s
      Preemptible:      %v
`, c.tpuHardware, c.preemptibleTPU)
	}
	fmt.Println()
	return askForConfirmation("OK to create your Cloud TPU resources with the above configuration?")
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

	// Determine if the APIs have been enabled and/or if there are instances running
	tpuInstance, tpuEnabled, err := c.tpu.OptionallyRetrieveInstance(false)
	gceInstance, _, err := c.gce.OptionallyRetrieveInstance(false)
	alreadyRunning := tpuInstance != nil && tpuInstance.IsRunning() && gceInstance != nil && gceInstance.IsRunning()

	if tpuEnabled {
		// TF Version check.
		err = c.ConfigureTFVersion()
		if err != nil {
			log.Print(err)
			return subcommands.ExitFailure
		}
	}

	if !alreadyRunning && !c.skipConfirmation {
		confirmed, err := c.confirmExecution(tpuEnabled)
		if err != nil {
			log.Print(err)
			return subcommands.ExitFailure
		}
		if !confirmed {
			color.Red("Canceling ctpu; no changes have been made.")
			fmt.Printf(`
Note: if there was a configuration issue, you can override the settings using
flags on the command line (e.g. --zone, --project, --name) for a given ctpu
execution. For the full list of flags, run: ctpu help up
`)
			return subcommands.ExitUsageError
		}
	}

	if !tpuEnabled {
		origTFVersion := c.tfVersion
		// Enable the API by attempting to retrieve the instance.
		_, err := c.tpu.Instance()
		if err != nil {
			log.Print(err)
			return subcommands.ExitFailure
		}
		// TF Version check.
		err = c.ConfigureTFVersion()
		if err != nil {
			log.Print(err)
			return subcommands.ExitFailure
		}
		if origTFVersion == "" {
			fmt.Printf("Selected TensorFlow version: %q. Launching instances...\n", c.tfVersion)
		}
	}

	tpu, err := c.launchInstances(ctx)
	if err != nil {
		fmt.Printf("%v\n", err)
		return subcommands.ExitFailure
	}

	if inOrg, err := c.rmg.IsProjectInGoogleOrg(); err == nil && inOrg && c.cfg.Environment == "devshell" {
		color.Red("WARNING: Attempting to ssh from Cloud Shell in Google org; this might not work!")
	}

	if c.forwardPorts {
		fmt.Println("About to ssh (with port forwarding enabled -- see docs for details)...")
	} else {
		fmt.Println("About to ssh...")
	}
	if err := c.cli.SSHToInstance(c.forwardPorts, c.forwardAgent, tpu); err != nil {
		log.Printf("Could not ssh to instance: %v\n", err)
		return subcommands.ExitFailure
	}
	return subcommands.ExitSuccess
}
