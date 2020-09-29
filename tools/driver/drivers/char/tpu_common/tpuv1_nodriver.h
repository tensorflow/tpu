/* SPDX-License-Identifier: GPL-2.0 */
/*
 * Copyright (C) 2020 Google LLC.
 */
#ifndef _DRIVERS_CHAR_TPU_COMMON_TPUV1_NODRIVER_H_
#define _DRIVERS_CHAR_TPU_COMMON_TPUV1_NODRIVER_H_ 
#include "linux/pci.h"
int tpuv1_nodriver_reset(struct pci_dev *pci_dev);
#endif
