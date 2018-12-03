// Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
	"sort"

	"flag"
	"github.com/google/subcommands"
	"github.com/tensorflow/tpu/tools/ctpu/config"
	"google.golang.org/api/tpu/v1alpha1"
)

// TPUSizeCP represents the interface required of the TPU control plane to run this subcommand.
type TPUSizeCP interface {
	// ListSizes lists available TPU sizes.
	ListSizes() ([]*tpu.AcceleratorType, error)
}

type tpuSizeCmd struct {
	cfg  *config.Config
	tpus TPUSizeCP
}

// TPUSizeCommand constructs the tpu-sizes subcommand.
func TPUSizeCommand(cfg *config.Config, tpus TPUSizeCP) subcommands.Command {
	return &tpuSizeCmd{cfg, tpus}
}

func (tpuSizeCmd) Name() string {
	return "tpu-sizes"
}

func (t *tpuSizeCmd) SetFlags(f *flag.FlagSet) {
	t.cfg.SetFlags(f)
}

func (tpuSizeCmd) Synopsis() string {
	return "queries for all available TPU sizes. Note: some sizes may only be available in certain locations."
}

func (tpuSizeCmd) Usage() string {
	return `ctpu tpu-sizes
`
}

func sortTpuSizes(sizes []*tpu.AcceleratorType) {
	sort.Slice(sizes, func(i, j int) bool {
		// First sort by version:
		if sizes[i].Type[1] != sizes[j].Type[1] {
			return sizes[i].Type < sizes[j].Type
		}
		// Then sort by length:
		if len(sizes[i].Type) != len(sizes[j].Type) {
			return len(sizes[i].Type) < len(sizes[j].Type)
		}
		// Then sort normally.
		return sizes[i].Type < sizes[j].Type
	})
}

func (t *tpuSizeCmd) Execute(ctx context.Context, flags *flag.FlagSet, args ...interface{}) subcommands.ExitStatus {
	err := t.cfg.Validate()
	if err != nil {
		log.Print(err)
		return subcommands.ExitFailure
	}

	types, err := t.tpus.ListSizes()
	if err != nil {
		log.Print(err)
		return subcommands.ExitFailure
	}
	if len(types) == 0 {
		fmt.Printf("No available Cloud TPU sizes..\n")
		return subcommands.ExitFailure
	}

	sortTpuSizes(types)

	fmt.Printf("Cloud TPU sizes available in %s:\n", t.cfg.Zone)
	for _, tpe := range types {
		fmt.Printf("\t%s\n", tpe.Type)
	}

	return subcommands.ExitSuccess
}
