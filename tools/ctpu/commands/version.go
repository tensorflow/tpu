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

	"flag"
	"github.com/google/subcommands"
)

type versionCmd struct {
	version string
}

// VersionCommand creates the version command.
func VersionCommand(version string) subcommands.Command {
	return &versionCmd{version: version}
}

func (versionCmd) Name() string {
	return "version"
}

func (versionCmd) SetFlags(f *flag.FlagSet) {}

func (versionCmd) Synopsis() string {
	return "prints out the ctpu version."
}

func (versionCmd) Usage() string {
	return `ctpu version
`
}

func (v *versionCmd) Execute(ctx context.Context, flags *flag.FlagSet, args ...interface{}) subcommands.ExitStatus {
	fmt.Printf("ctpu version: %s\n", v.version)
	return subcommands.ExitSuccess
}
