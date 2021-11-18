/* SPDX-License-Identifier: GPL-2.0 */
/*
 * Copyright (C) 2021 Google LLC.
 */
#include "drivers/asic_sw/asic_sw_clock.h"
#include <linux/export.h>
#include <linux/jiffies.h>
#include <linux/sched.h>
void asic_sw_sleep_for_msecs(uint64 duration_msecs)
{
 set_current_state(TASK_UNINTERRUPTIBLE);
 schedule_timeout(msecs_to_jiffies(duration_msecs));
}
EXPORT_SYMBOL(asic_sw_sleep_for_msecs);
