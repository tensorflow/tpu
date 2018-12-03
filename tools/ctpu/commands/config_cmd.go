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
)

// ConfigGcloudCLI encapsulates interaction with the gcloud CLI toolchain.
type ConfigGcloudCLI interface {
	// IsGcloudInstalled returns true if gcloud is installed.
	IsGcloudInstalled() bool
	// PrintInstallInstructions prints install instructions to the console.
	PrintInstallInstructions()
}

type configCmd struct {
	cfg  *config.Config
	cli  ConfigGcloudCLI
	full bool
}

// ConfigCommand creates the config subcommand.
func ConfigCommand(cfg *config.Config, cli ConfigGcloudCLI) subcommands.Command {
	return &configCmd{cfg, cli, false}
}

func (configCmd) Name() string {
	return "print-config"
}

func (c *configCmd) SetFlags(f *flag.FlagSet) {
	c.cfg.SetFlags(f) // Allow users to specify cfg flags either before or after the subcommand name.
	f.BoolVar(&c.full, "full", false, "Print the full configuration.")
}

func (configCmd) Synopsis() string {
	return "prints out configuration."
}
func (configCmd) Usage() string {
	return `ctpu print-config
`
}

type configCmdAlias struct {
	configCmd
}

// ConfigCommandAlias creates an alias to the config command with a shorter name.
func ConfigCommandAlias(cfg *config.Config, cli ConfigGcloudCLI) subcommands.Command {
	return &configCmdAlias{configCmd{cfg: cfg, cli: cli}}
}
func (configCmdAlias) Name() string     { return "cfg" }
func (configCmdAlias) Synopsis() string { return "alias to ctpu print-config" }
func (configCmdAlias) Usage() string    { return "ctpu cfg\n" }

func (c *configCmd) Execute(ctx context.Context, flags *flag.FlagSet, args ...interface{}) subcommands.ExitStatus {
	err := c.cfg.Validate()
	if err != nil {
		log.Print(err)
		return subcommands.ExitFailure
	}

	fmt.Printf("ctpu configuration:\n\tname: %s\n\tproject: %s\n\tzone: %s\n",
		c.cfg.FlockName, c.cfg.Project, c.cfg.Zone)
	if c.full {
		fmt.Printf("\tenvironment: %s\n\tactive gcloud sdk configuration: %s\n", c.cfg.Environment, c.cfg.ActiveConfiguration)
	}
	fmt.Printf("If you would like to change the configuration for a single command invocation, please use the command line flags.\n")
	if c.cfg.Environment == "gcloud" {
		fmt.Printf("If you would like to change the configuration generally, see `gcloud config configurations`.\n")
	}

	if !c.cli.IsGcloudInstalled() {
		c.cli.PrintInstallInstructions()
		return subcommands.ExitFailure
	}
	return subcommands.ExitSuccess
}
