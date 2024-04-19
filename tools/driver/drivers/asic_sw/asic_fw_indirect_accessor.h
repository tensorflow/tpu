/* SPDX-License-Identifier: GPL-2.0 */
/*
 * Copyright (C) 2021 Google LLC.
 */
#ifndef _DRIVERS_ASIC_SW_ASIC_FW_INDIRECT_ACCESSOR_H_
#define _DRIVERS_ASIC_SW_ASIC_FW_INDIRECT_ACCESSOR_H_ 
#include "drivers/gasket/gasket_types.h"
enum asic_fw_indirect_accessor_status_status_value {
 kAsicFwIndirectAccessorStatusStatusValueOk = 0,
 kAsicFwIndirectAccessorStatusStatusValueFailed = 1,
 kAsicFwIndirectAccessorStatusStatusValueInvalid = 2
};
typedef enum asic_fw_indirect_accessor_status_status_value
 asic_fw_indirect_accessor_status_status_value;
static inline uint64_t
asic_fw_indirect_accessor_version_version(const uint64_t reg_value)
{
 return (uint64_t)((((reg_value >> 0) & 0xffffffffffffffffULL) << 0));
}
static inline int
set_asic_fw_indirect_accessor_version_version(uint64_t *reg_value,
           uint64_t value)
{
 if ((uint64_t)value < 0x0ULL || (uint64_t)value > 0xffffffffffffffffULL)
  return 1;
 (*reg_value) = ((*reg_value) & ~((0xffffffffffffffffULL) << 0)) |
         (((value >> 0) & (0xffffffffffffffffULL)) << 0);
 return 0;
}
static inline uint64_t
asic_fw_indirect_accessor_address_address(const uint64_t reg_value)
{
 return (uint64_t)((((reg_value >> 0) & 0xffffffffffffffffULL) << 0));
}
static inline int
set_asic_fw_indirect_accessor_address_address(uint64_t *reg_value,
           uint64_t value)
{
 if ((uint64_t)value < 0x0ULL || (uint64_t)value > 0xffffffffffffffffULL)
  return 1;
 (*reg_value) = ((*reg_value) & ~((0xffffffffffffffffULL) << 0)) |
         (((value >> 0) & (0xffffffffffffffffULL)) << 0);
 return 0;
}
static inline uint8_t
asic_fw_indirect_accessor_control_write(const uint64_t reg_value)
{
 return (uint8_t)((((reg_value >> 0) & 0x1ULL) << 0));
}
static inline int
set_asic_fw_indirect_accessor_control_write(uint64_t *reg_value, uint8_t value)
{
 if ((uint64_t)value < 0x0ULL || (uint64_t)value > 0x1ULL)
  return 1;
 (*reg_value) = ((*reg_value) & ~((0x1ULL) << 0)) |
         (((value >> 0) & (0x1ULL)) << 0);
 return 0;
}
static inline uint8_t
asic_fw_indirect_accessor_control_read(const uint64_t reg_value)
{
 return (uint8_t)((((reg_value >> 1) & 0x1ULL) << 0));
}
static inline int
set_asic_fw_indirect_accessor_control_read(uint64_t *reg_value, uint8_t value)
{
 if ((uint64_t)value < 0x0ULL || (uint64_t)value > 0x1ULL)
  return 1;
 (*reg_value) = ((*reg_value) & ~((0x1ULL) << 1)) |
         (((value >> 0) & (0x1ULL)) << 1);
 return 0;
}
static inline bool
asic_fw_indirect_accessor_status_status_value_is_valid(int value)
{
 if (value == 0) {
  return true;
 }
 if (value == 1) {
  return true;
 }
 if (value == 2) {
  return true;
 }
 return false;
}
static inline const char *asic_fw_indirect_accessor_status_status_value_name(
 asic_fw_indirect_accessor_status_status_value value)
{
 if (value == 0) {
  return "OK";
 }
 if (value == 1) {
  return "FAILED";
 }
 if (value == 2) {
  return "INVALID";
 }
 return "UNKNOWN VALUE";
}
static inline asic_fw_indirect_accessor_status_status_value
asic_fw_indirect_accessor_status_status(const uint64_t reg_value)
{
 return (asic_fw_indirect_accessor_status_status_value)((
  ((reg_value >> 0) & 0xffULL) << 0));
}
static inline int set_asic_fw_indirect_accessor_status_status(
 uint64_t *reg_value,
 asic_fw_indirect_accessor_status_status_value value)
{
 if ((uint64_t)value < 0x0ULL || (uint64_t)value > 0xffULL)
  return 1;
 (*reg_value) = ((*reg_value) & ~((0xffULL) << 0)) |
         (((value >> 0) & (0xffULL)) << 0);
 return 0;
}
static inline uint8_t
asic_fw_indirect_accessor_status_chip_specific_status(const uint64_t reg_value)
{
 return (uint8_t)((((reg_value >> 8) & 0xffULL) << 0));
}
static inline int
set_asic_fw_indirect_accessor_status_chip_specific_status(uint64_t *reg_value,
         uint8_t value)
{
 if ((uint64_t)value < 0x0ULL || (uint64_t)value > 0xffULL)
  return 1;
 (*reg_value) = ((*reg_value) & ~((0xffULL) << 8)) |
         (((value >> 0) & (0xffULL)) << 8);
 return 0;
}
static inline uint64_t
asic_fw_indirect_accessor_value_value(const uint64_t reg_value)
{
 return (uint64_t)((((reg_value >> 0) & 0xffffffffffffffffULL) << 0));
}
static inline int set_asic_fw_indirect_accessor_value_value(uint64_t *reg_value,
           uint64_t value)
{
 if ((uint64_t)value < 0x0ULL || (uint64_t)value > 0xffffffffffffffffULL)
  return 1;
 (*reg_value) = ((*reg_value) & ~((0xffffffffffffffffULL) << 0)) |
         (((value >> 0) & (0xffffffffffffffffULL)) << 0);
 return 0;
}
#endif
