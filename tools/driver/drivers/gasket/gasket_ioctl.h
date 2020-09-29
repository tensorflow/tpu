/* SPDX-License-Identifier: GPL-2.0 */
/*
 * Copyright (C) 2020 Google LLC.
 */
#ifndef __GASKET_IOCTL_H__
#define __GASKET_IOCTL_H__ 
#include "gasket_core.h"
long gasket_handle_ioctl(struct file *filp, uint cmd, ulong arg);
long gasket_is_supported_ioctl(uint cmd);
#endif
