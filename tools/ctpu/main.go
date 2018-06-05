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

// Package main contains the ctpu main program.
package main

import (
	"fmt"
	"os"

	"flag"
	// context is used to cancel outstanding requests
	"context"
	"github.com/google/subcommands"
	"github.com/tensorflow/tpu/tools/ctpu/commands"
	"github.com/tensorflow/tpu/tools/ctpu/config"
	"github.com/tensorflow/tpu/tools/ctpu/ctrl"
)

const version = "1.1"

var logRequests bool

func check(context string, err error) {
	if err != nil {
		fmt.Printf("Error encountered while %s: %s\n", context, err)
		os.Exit(1)
	}
}

func init() {
	flag.BoolVar(&logRequests, "log-http", false,
		`When debugging the ctpu tool, it can be helpful to see the full content of
        all HTTP request-response pairs. Set this flag to true in order to print out
        the requests.`)
}

func main() {
	cfg, err := config.FromEnv()
	check("creating configuration", err)
	cfg.SetFlags(flag.CommandLine)
	flag.Parse()

	ctx := context.Background()
	ctrls, err := ctrl.New(ctx, cfg, version, logRequests)
	check("creating API clients", err)

	subcommands.ImportantFlag("name")
	subcommands.ImportantFlag("project")
	subcommands.ImportantFlag("zone")

	subcommands.Register(commands.UpCommand(cfg, ctrls.TPU, ctrls.GCE, ctrls.ResourceManagement, ctrls.CLI), "")
	subcommands.Register(commands.PauseCommand(cfg, ctrls.TPU, ctrls.GCE), "")
	subcommands.Register(commands.DeleteCommand(cfg, ctrls.TPU, ctrls.GCE), "")
	subcommands.Register(commands.StatusCommand(cfg, ctrls.TPU, ctrls.GCE), "")
	subcommands.Register(commands.RestartCommand(cfg, ctrls.TPU), "")

	subcommands.Register(commands.ConfigCommand(cfg, ctrls.CLI), "configuration")
	subcommands.Register(commands.VersionCommand(version), "configuration")
	subcommands.Register(commands.ListCommand(cfg, ctrls.TPU, ctrls.GCE), "configuration")
	subcommands.Register(commands.TFVersionsCommand(cfg, ctrls.TPU), "configuration")
	subcommands.Register(commands.TPULocationsCommand(cfg, ctrls.TPU), "configuration")
	subcommands.Register(commands.QuotaCommand(cfg), "configuration")

	subcommands.Register(commands.PauseCommandAlias(cfg, ctrls.TPU, ctrls.GCE), "aliases for other commands")
	subcommands.Register(commands.DeleteCommandAlias(cfg, ctrls.TPU, ctrls.GCE), "aliases for other commands")
	subcommands.Register(commands.StatusCommandAlias(cfg, ctrls.TPU, ctrls.GCE), "aliases for other commands")
	subcommands.Register(commands.ListCommandAlias(cfg, ctrls.TPU, ctrls.GCE), "aliases for other commands")
	subcommands.Register(commands.ConfigCommandAlias(cfg, ctrls.CLI), "aliases for other commands")

	subcommands.Register(subcommands.HelpCommand(), "usage")
	subcommands.Register(subcommands.FlagsCommand(), "usage")
	subcommands.Register(subcommands.CommandsCommand(), "usage")

	os.Exit(int(subcommands.Execute(ctx)))
}
