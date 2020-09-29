/* SPDX-License-Identifier: GPL-2.0 */
/*
 * Copyright (C) 2020 Google LLC.
 */
#ifndef _LINUX_GOOGLE_ASIC_FW_DEVICE_OWNER_ACCESSOR_H_
#define _LINUX_GOOGLE_ASIC_FW_DEVICE_OWNER_ACCESSOR_H_ 
#include <linux/types.h>
inline unsigned long asic_fw_device_owner_value(const unsigned long *reg_value)
{
 return (unsigned long)(((((*reg_value) >> 0) & 0xffffffffffffffffULL)
    << 0));
}
inline int set_asic_fw_device_owner_value(unsigned long *reg_value,
       unsigned long value)
{
 if (value & ~(0xffffffffffffffffULL))
  return 1;
 (*reg_value) = ((*reg_value) & ~((0xffffffffffffffffULL) << 0)) |
         (((value >> 0) & (0xffffffffffffffffULL)) << 0);
 return 0;
}
#endif
