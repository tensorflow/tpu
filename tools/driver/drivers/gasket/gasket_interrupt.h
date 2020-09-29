/* SPDX-License-Identifier: GPL-2.0 */
/*
 * Copyright (C) 2020 Google LLC.
 */
#ifndef __GASKET_INTERRUPT_H__
#define __GASKET_INTERRUPT_H__ 
#include <linux/eventfd.h>
#include <linux/pci.h>
#include "gasket_core.h"
struct gasket_interrupt_data;
int legacy_gasket_interrupt_init(struct gasket_dev *gasket_dev,
 const char *name, const struct legacy_gasket_interrupt_desc *interrupts,
 int num_interrupts, int pack_width, int bar_index);
int legacy_gasket_interrupt_setup(struct gasket_dev *gasket_dev);
int gasket_interrupt_init(struct gasket_dev *gasket_dev,
 const char *name, const struct gasket_interrupt_desc *interrupts,
 int num_interrupts, int num_msix_interrupts);
void gasket_interrupt_cleanup(struct gasket_dev *gasket_dev);
int gasket_interrupt_reinit(struct gasket_dev *gasket_dev);
int gasket_interrupt_reset_counts(struct gasket_dev *gasket_dev);
int legacy_gasket_interrupt_set_eventfd(
 struct gasket_interrupt_data *interrupt_data, int interrupt,
 int event_fd);
int legacy_gasket_interrupt_clear_eventfd(
 struct gasket_interrupt_data *interrupt_data, int interrupt);
int gasket_interrupt_system_status(struct gasket_dev *gasket_dev);
struct eventfd_ctx **gasket_interrupt_get_eventfd_ctxs(
 struct gasket_interrupt_data *interrupt_data);
int gasket_interrupt_register_mapping(struct gasket_dev *gasket_dev,
 int interrupt, int event_fd, int bar_index, u64 reg);
int gasket_interrupt_unregister_mapping(struct gasket_dev *gasket_dev,
 int interrupt);
#endif
