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
	"github.com/google/subcommands"
	"github.com/tensorflow/tpu/tools/ctpu/config"
	"google.golang.org/api/tpu/v1alpha1"
)

// TFVersionsCP lists the available TensorFlow versions
type TFVersionsCP interface {
	// ListVersions returns the available TPU TensorFlow versions.
	ListVersions() ([]*tpu.TensorFlowVersion, error)
}

type tfVersionsCmd struct {
	cfg      *config.Config
	versions TFVersionsCP
}

// TFVersionsCommand creates the 'tf-versions' command.
func TFVersionsCommand(cfg *config.Config, versions TFVersionsCP) subcommands.Command {
	return &tfVersionsCmd{cfg, versions}
}

func (tfVersionsCmd) Name() string {
	return "tf-versions"
}

func (tfVersionsCmd) SetFlags(f *flag.FlagSet) {}

func (tfVersionsCmd) Synopsis() string {
	return "queries the control plane for the available TF versions."
}

func (tfVersionsCmd) Usage() string {
	return "ctpu tf-versions\n"
}

func (c *tfVersionsCmd) Execute(ctx context.Context, flags *flag.FlagSet, args ...interface{}) subcommands.ExitStatus {
	err := c.cfg.Validate()
	if err != nil {
		log.Print(err)
		return subcommands.ExitFailure
	}

	versions, err := c.versions.ListVersions()
	if err != nil {
		log.Print(err)
		return subcommands.ExitFailure
	}

	fmt.Printf("Cloud TPU TensorFlow Versions:\n")

	parsedVersions := make([]parsedVersion, 0, len(versions))
	for _, version := range versions {
		parsed, err := parseVersion(version.Version)
		if err != nil {
			log.Print(err)
			return subcommands.ExitFailure
		}
		parsedVersions = append(parsedVersions, parsed)
	}
	sortedVersions := sortedParsedVersions(parsedVersions)
	sort.Sort(sortedVersions)
	defaultVersion, err := sortedVersions.LatestStableRelease()
	if err != nil {
		log.Printf("WARNING: Could not determine latest stable release: %v", err)
		// Continue on with execution.
	}

	for _, version := range parsedVersions {
		annotation := ""
		if version.versionString() == defaultVersion {
			annotation = "\t(default version)"
		}
		fmt.Printf("\t%s%s\n", version.versionString(), annotation)
	}

	return subcommands.ExitSuccess
}
