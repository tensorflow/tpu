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
	"google.golang.org/api/tpu/v1alpha1"
)

type tpuLocationsCmd struct{}

// TPULocationsCommand creates the tpu-locations command.
func TPULocationsCommand() subcommands.Command {
	return &tpuLocationsCmd{}
}

func (tpuLocationsCmd) Name() string {
	return "tpu-locations"
}

func (tpuLocationsCmd) SetFlags(f *flag.FlagSet) {}

func (tpuLocationsCmd) Synopsis() string {
	return "queries for all locations with TPUs available."
}

func (tpuLocationsCmd) Usage() string {
	return `ctpu tpu-locations
`
}

func sortLocations(locations []*tpu.Location) {
	sort.Slice(locations, func(i, j int) bool { return locations[i].LocationId < locations[j].LocationId })
}

func (tpuLocationsCmd) Execute(ctx context.Context, flags *flag.FlagSet, args ...interface{}) subcommands.ExitStatus {
	libs, err := parseArgs(args)
	if err != nil {
		log.Print(err)
		return subcommands.ExitFailure
	}

	locations, err := libs.tpu.ListLocations()
	if err != nil {
		log.Print(err)
		return subcommands.ExitFailure
	}
	if len(locations) == 0 {
		fmt.Printf("No available Cloud TPU locations.\n")
		return subcommands.ExitFailure
	}

	sortLocations(locations)

	fmt.Printf("Cloud TPU Locations:\n")
	for _, loc := range locations {
		fmt.Printf("\t%s\n", loc.LocationId)
	}

	return subcommands.ExitSuccess
}
