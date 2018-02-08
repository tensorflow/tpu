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

const version = "0.4-dev"

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
	config.RegisterFlags()
	flag.Parse()

	subcommands.ImportantFlag("name")
	subcommands.ImportantFlag("project")
	subcommands.ImportantFlag("zone")

	subcommands.Register(commands.UpCommand(), "")
	subcommands.Register(commands.PauseCommand(), "")
	subcommands.Register(commands.DeleteCommand(), "")
	subcommands.Register(commands.StatusCommand(), "")

	subcommands.Register(commands.ConfigCommand(), "configuration")
	subcommands.Register(commands.VersionCommand(version), "configuration")
	subcommands.Register(commands.ListCommand(), "configuration")
	subcommands.Register(commands.TFVersionsCommand(), "configuration")
	subcommands.Register(commands.TPULocationsCommand(), "configuration")

	subcommands.Register(subcommands.HelpCommand(), "usage")
	subcommands.Register(subcommands.FlagsCommand(), "usage")
	subcommands.Register(subcommands.CommandsCommand(), "usage")

	ctx := context.Background()
	cfg, err := config.NewConfig()
	check("creating configuration", err)
	ctrls, err := ctrl.New(ctx, cfg, version, logRequests)
	check("creating API clients", err)

	os.Exit(int(subcommands.Execute(ctx, cfg, ctrls)))
}
