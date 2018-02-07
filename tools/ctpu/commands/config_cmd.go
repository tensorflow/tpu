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

	"context"
	"flag"
	"github.com/google/subcommands"
)

type configCmd struct{}

// ConfigCommand creates the config subcommand.
func ConfigCommand() subcommands.Command {
	return &configCmd{}
}

func (configCmd) Name() string {
	return "config"
}

func (c *configCmd) SetFlags(f *flag.FlagSet) {
}

func (configCmd) Synopsis() string {
	return "prints out configuration."
}
func (configCmd) Usage() string {
	return `ctpu config
`
}

func (c *configCmd) Execute(ctx context.Context, flags *flag.FlagSet, args ...interface{}) subcommands.ExitStatus {
	libs, err := parseArgs(args)
	if err != nil {
		log.Print(err)
		return subcommands.ExitFailure
	}
	fmt.Printf("ctpu configuration:\n\tname: %s\n\tproject: %s\n\tzone: %s\n",
		libs.cfg.FlockName(), libs.cfg.Project(), libs.cfg.Zone())
	fmt.Printf("If you would like to change the configuration for a single command invocation, please use the command line flags. If you would like to change the configuration generally, see `gcloud config configurations`.\n")

	if !libs.cli.IsGcloudInstalled() {
		libs.cli.PrintInstallInstructions()
		return subcommands.ExitFailure
	}
	return subcommands.ExitSuccess
}
