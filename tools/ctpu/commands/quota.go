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

type quotaCmd struct{}

// QuotaCommand creates the quota command.
func QuotaCommand() subcommands.Command {
	return &quotaCmd{}
}

func (quotaCmd) Name() string {
	return "quota"
}

func (quotaCmd) SetFlags(f *flag.FlagSet) {}

func (quotaCmd) Synopsis() string {
	return "prints URL where quota can be seen"
}

func (quotaCmd) Usage() string {
	return `ctpu quota
`
}

func (quotaCmd) Execute(ctx context.Context, flags *flag.FlagSet, args ...interface{}) subcommands.ExitStatus {
	libs, err := parseArgs(args)
	if err != nil {
		log.Printf("%v\n", err)
		return subcommands.ExitFailure
	}

	fmt.Printf("Quotas are not available within ctpu. Head over to:\n\thttps://console.cloud.google.com/iam-admin/quotas?project=%s&service=tpu.googleapis.com\n", libs.cfg.Project())

	return subcommands.ExitSuccess
}
