/* SPDX-License-Identifier: GPL-2.0 */
/*
 * Copyright (C) 2021 Google LLC.
 */
#include <linux/device.h>
#include <linux/pci.h>
#include <linux/printk.h>
#ifndef _GASKET_LOGGING_H_
#define _GASKET_LOGGING_H_ 
#define gasket_dev_log(level,gasket_dev,format,arg...) \
 { if ((gasket_dev)->parent) { \
  dev_##level##_ratelimited(&(gasket_dev)->accel_dev.dev, \
       "(c%d) %s: " format "\n", \
       (gasket_dev)->clone_index, __func__, ##arg); \
 } else { \
  dev_##level##_ratelimited(&(gasket_dev)->accel_dev.dev, \
       "%s: " format "\n", __func__, ##arg); \
 } }
#define gasket_nodev_log(level,format,arg...) \
 pr_##level##_ratelimited("gasket: %s: " format "\n", __func__, ##arg)
#define gasket_nodev_debug(format,arg...) \
 gasket_nodev_log(debug, format, ##arg)
#define gasket_nodev_info(format,arg...) gasket_nodev_log(info, format, ##arg)
#define gasket_nodev_warn(format,arg...) gasket_nodev_log(warn, format, ##arg)
#define gasket_nodev_error(format,arg...) gasket_nodev_log(err, format, ##arg)
#define gasket_log_debug(gasket_dev,format,arg...) \
 gasket_dev_log(dbg, (gasket_dev), format, ##arg)
#define gasket_log_info(gasket_dev,format,arg...) \
 gasket_dev_log(info, (gasket_dev), format, ##arg)
#define gasket_log_warn(gasket_dev,format,arg...) \
 gasket_dev_log(warn, (gasket_dev), format, ##arg)
#define gasket_log_error(gasket_dev,format,arg...) \
 gasket_dev_log(err, (gasket_dev), format, ##arg)
#endif
