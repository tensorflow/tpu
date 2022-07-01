/* SPDX-License-Identifier: GPL-2.0 */
/*
 * Copyright (C) 2021 Google LLC.
 */
#ifndef _DRIVERS_CHAR_TPU_COMMON_TPU_V4_COMMON_TPU_V4_COMMON_USERSPACE_LST_PORT_INDIRECT_ACCESSOR_H_
#define _DRIVERS_CHAR_TPU_COMMON_TPU_V4_COMMON_TPU_V4_COMMON_USERSPACE_LST_PORT_INDIRECT_ACCESSOR_H_ 
#include "drivers/gasket/gasket_types.h"
enum tpuv4common_lock_lock_value {
 kTpuv4commonLockLockValueUnlocked = 0,
 kTpuv4commonLockLockValueLocked = 1
};
typedef enum tpuv4common_lock_lock_value tpuv4common_lock_lock_value;
enum tpuv4common_data_link_layer_request_request_value {
 kTpuv4commonDataLinkLayerRequestRequestValueNoRequest = 0,
 kTpuv4commonDataLinkLayerRequestRequestValueGoUp = 1,
 kTpuv4commonDataLinkLayerRequestRequestValueGoDown = 2
};
typedef enum tpuv4common_data_link_layer_request_request_value
 tpuv4common_data_link_layer_request_request_value;
enum tpuv4common_data_link_layer_status_status_value {
 kTpuv4commonDataLinkLayerStatusStatusValueDown = 1,
 kTpuv4commonDataLinkLayerStatusStatusValueGoingUp = 2,
 kTpuv4commonDataLinkLayerStatusStatusValueUp = 3
};
typedef enum tpuv4common_data_link_layer_status_status_value
 tpuv4common_data_link_layer_status_status_value;
enum tpuv4common_unused_register_two_config_status_value {
 kTpuv4commonIciConnectorInfoConfigStatusValueUnknown = 0,
 kTpuv4commonIciConnectorInfoConfigStatusValueConfigValid = 1,
 kTpuv4commonIciConnectorInfoConfigStatusValueConfigInvalid = 2
};
typedef enum tpuv4common_unused_register_two_config_status_value
 tpuv4common_unused_register_two_config_status_value;
enum tpuv4common_unused_register_two_connector_id_value {
 kTpuv4commonIciConnectorInfoConnectorIdValueIci0 = 0,
 kTpuv4commonIciConnectorInfoConnectorIdValueIci1 = 1,
 kTpuv4commonIciConnectorInfoConnectorIdValueIci2 = 2,
 kTpuv4commonIciConnectorInfoConnectorIdValueIci3 = 3,
 kTpuv4commonIciConnectorInfoConnectorIdValueIci4 = 4,
 kTpuv4commonIciConnectorInfoConnectorIdValueIci5 = 5,
 kTpuv4commonIciConnectorInfoConnectorIdValueIci6 = 6,
 kTpuv4commonIciConnectorInfoConnectorIdValueIci7 = 7,
 kTpuv4commonIciConnectorInfoConnectorIdValueIci8 = 8,
 kTpuv4commonIciConnectorInfoConnectorIdValueIci9 = 9,
 kTpuv4commonIciConnectorInfoConnectorIdValueIci10 = 10,
 kTpuv4commonIciConnectorInfoConnectorIdValueIci11 = 11,
 kTpuv4commonIciConnectorInfoConnectorIdValueIci12 = 12,
 kTpuv4commonIciConnectorInfoConnectorIdValueIci13 = 13,
 kTpuv4commonIciConnectorInfoConnectorIdValueIci14 = 14,
 kTpuv4commonIciConnectorInfoConnectorIdValueIci15 = 15
};
typedef enum tpuv4common_unused_register_two_connector_id_value
 tpuv4common_unused_register_two_connector_id_value;
enum tpuv4common_unused_register_two_connector_type_value {
 kTpuv4commonIciConnectorInfoConnectorTypeValueInternal = 0,
 kTpuv4commonIciConnectorInfoConnectorTypeValueCopper = 1,
 kTpuv4commonIciConnectorInfoConnectorTypeValueOptics = 2
};
typedef enum tpuv4common_unused_register_two_connector_type_value
 tpuv4common_unused_register_two_connector_type_value;
enum tpuv4common_unused_register_two_orientation_value {
 kTpuv4commonIciConnectorInfoOrientationValueUnknown = 0,
 kTpuv4commonIciConnectorInfoOrientationValueX = 1,
 kTpuv4commonIciConnectorInfoOrientationValueY = 2,
 kTpuv4commonIciConnectorInfoOrientationValueZ = 3
};
typedef enum tpuv4common_unused_register_two_orientation_value
 tpuv4common_unused_register_two_orientation_value;
enum tpuv4common_unused_register_two_polarity_value {
 kTpuv4commonIciConnectorInfoPolarityValueUnknown = 0,
 kTpuv4commonIciConnectorInfoPolarityValuePositive = 1,
 kTpuv4commonIciConnectorInfoPolarityValueNegative = 2
};
typedef enum tpuv4common_unused_register_two_polarity_value
 tpuv4common_unused_register_two_polarity_value;
enum tpuv4common_physical_layer_state_value_value {
 kTpuv4commonPhysicalLayerStateValueValueDisabled = 0,
 kTpuv4commonPhysicalLayerStateValueValueDown = 1,
 kTpuv4commonPhysicalLayerStateValueValueUp = 2
};
typedef enum tpuv4common_physical_layer_state_value_value
 tpuv4common_physical_layer_state_value_value;
enum tpuv4common_unused_register_one_cable_type_value {
 kTpuv4commonIciConnectorGoogleInfoCableTypeValueUnknown = 0,
 kTpuv4commonIciConnectorGoogleInfoCableTypeValueStraight = 1,
 kTpuv4commonIciConnectorGoogleInfoCableTypeValueY = 2,
 kTpuv4commonIciConnectorGoogleInfoCableTypeValueQuad = 3
};
typedef enum tpuv4common_unused_register_one_cable_type_value
 tpuv4common_unused_register_one_cable_type_value;
enum tpuv4common_unused_register_one_connector_end_value {
 kTpuv4commonIciConnectorGoogleInfoConnectorEndValueUnknown = 0,
 kTpuv4commonIciConnectorGoogleInfoConnectorEndValueEnd1 = 1,
 kTpuv4commonIciConnectorGoogleInfoConnectorEndValueEnd2 = 2
};
typedef enum tpuv4common_unused_register_one_connector_end_value
 tpuv4common_unused_register_one_connector_end_value;
static inline bool tpuv4common_lock_lock_value_is_valid(int value)
{
 if (value == 0) {
  return true;
 }
 if (value == 1) {
  return true;
 }
 return false;
}
static inline const char *tpuv4common_lock_lock_value_name(tpuv4common_lock_lock_value value)
{
 if (value == 0) {
  return "UNLOCKED";
 }
 if (value == 1) {
  return "LOCKED";
 }
 return "UNKNOWN VALUE";
}
static inline tpuv4common_lock_lock_value tpuv4common_lock_lock(const uint64 reg_value)
{
 return (tpuv4common_lock_lock_value)(
  (((reg_value >> 0) & 0xffffffffffffffffULL) << 0));
}
static inline int set_tpuv4common_lock_lock(uint64 *reg_value,
        tpuv4common_lock_lock_value value)
{
 if (value & ~(0xffffffffffffffffULL))
  return 1;
 (*reg_value) = ((*reg_value) & ~((0xffffffffffffffffULL) << 0)) |
         (((value >> 0) & (0xffffffffffffffffULL)) << 0);
 return 0;
}
static inline uint64 tpuv4common_to_mirror_value(const uint64 reg_value)
{
 return (uint64)((((reg_value >> 0) & 0xffffffffffffffffULL) << 0));
}
static inline int set_tpuv4common_to_mirror_value(uint64 *reg_value, uint64 value)
{
 if (value & ~(0xffffffffffffffffULL))
  return 1;
 (*reg_value) = ((*reg_value) & ~((0xffffffffffffffffULL) << 0)) |
         (((value >> 0) & (0xffffffffffffffffULL)) << 0);
 return 0;
}
static inline bool tpuv4common_data_link_layer_request_request_value_is_valid(int value)
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
static inline const char *tpuv4common_data_link_layer_request_request_value_name(
 tpuv4common_data_link_layer_request_request_value value)
{
 if (value == 0) {
  return "NO_REQUEST";
 }
 if (value == 1) {
  return "GO_UP";
 }
 if (value == 2) {
  return "GO_DOWN";
 }
 return "UNKNOWN VALUE";
}
static inline tpuv4common_data_link_layer_request_request_value
tpuv4common_data_link_layer_request_request(const uint64 reg_value)
{
 return (tpuv4common_data_link_layer_request_request_value)(
  (((reg_value >> 0) & 0xffffffffffffffffULL) << 0));
}
static inline int set_tpuv4common_data_link_layer_request_request(
 uint64 *reg_value, tpuv4common_data_link_layer_request_request_value value)
{
 if (value & ~(0xffffffffffffffffULL))
  return 1;
 (*reg_value) = ((*reg_value) & ~((0xffffffffffffffffULL) << 0)) |
         (((value >> 0) & (0xffffffffffffffffULL)) << 0);
 return 0;
}
static inline bool tpuv4common_data_link_layer_status_status_value_is_valid(int value)
{
 if (value == 1) {
  return true;
 }
 if (value == 2) {
  return true;
 }
 if (value == 3) {
  return true;
 }
 return false;
}
static inline const char *tpuv4common_data_link_layer_status_status_value_name(
 tpuv4common_data_link_layer_status_status_value value)
{
 if (value == 1) {
  return "DOWN";
 }
 if (value == 2) {
  return "GOING_UP";
 }
 if (value == 3) {
  return "UP";
 }
 return "UNKNOWN VALUE";
}
static inline tpuv4common_data_link_layer_status_status_value
tpuv4common_data_link_layer_status_status(const uint64 reg_value)
{
 return (tpuv4common_data_link_layer_status_status_value)(
  (((reg_value >> 0) & 0xffffffffffffffffULL) << 0));
}
static inline int set_tpuv4common_data_link_layer_status_status(
 uint64 *reg_value, tpuv4common_data_link_layer_status_status_value value)
{
 if (value & ~(0xffffffffffffffffULL))
  return 1;
 (*reg_value) = ((*reg_value) & ~((0xffffffffffffffffULL) << 0)) |
         (((value >> 0) & (0xffffffffffffffffULL)) << 0);
 return 0;
}
static inline bool
tpuv4common_unused_register_two_config_status_value_is_valid(int value)
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
static inline const char *tpuv4common_unused_register_two_config_status_value_name(
 tpuv4common_unused_register_two_config_status_value value)
{
 if (value == 0) {
  return "UNKNOWN";
 }
 if (value == 1) {
  return "CONFIG_VALID";
 }
 if (value == 2) {
  return "CONFIG_INVALID";
 }
 return "UNKNOWN VALUE";
}
static inline bool tpuv4common_unused_register_two_connector_id_value_is_valid(int value)
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
 if (value == 3) {
  return true;
 }
 if (value == 4) {
  return true;
 }
 if (value == 5) {
  return true;
 }
 if (value == 6) {
  return true;
 }
 if (value == 7) {
  return true;
 }
 if (value == 8) {
  return true;
 }
 if (value == 9) {
  return true;
 }
 if (value == 10) {
  return true;
 }
 if (value == 11) {
  return true;
 }
 if (value == 12) {
  return true;
 }
 if (value == 13) {
  return true;
 }
 if (value == 14) {
  return true;
 }
 if (value == 15) {
  return true;
 }
 return false;
}
static inline const char *tpuv4common_unused_register_two_connector_id_value_name(
 tpuv4common_unused_register_two_connector_id_value value)
{
 if (value == 0) {
  return "ICI0";
 }
 if (value == 1) {
  return "ICI1";
 }
 if (value == 2) {
  return "ICI2";
 }
 if (value == 3) {
  return "ICI3";
 }
 if (value == 4) {
  return "ICI4";
 }
 if (value == 5) {
  return "ICI5";
 }
 if (value == 6) {
  return "ICI6";
 }
 if (value == 7) {
  return "ICI7";
 }
 if (value == 8) {
  return "ICI8";
 }
 if (value == 9) {
  return "ICI9";
 }
 if (value == 10) {
  return "ICI10";
 }
 if (value == 11) {
  return "ICI11";
 }
 if (value == 12) {
  return "ICI12";
 }
 if (value == 13) {
  return "ICI13";
 }
 if (value == 14) {
  return "ICI14";
 }
 if (value == 15) {
  return "ICI15";
 }
 return "UNKNOWN VALUE";
}
static inline bool
tpuv4common_unused_register_two_connector_type_value_is_valid(int value)
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
static inline const char *tpuv4common_unused_register_two_connector_type_value_name(
 tpuv4common_unused_register_two_connector_type_value value)
{
 if (value == 0) {
  return "INTERNAL";
 }
 if (value == 1) {
  return "COPPER";
 }
 if (value == 2) {
  return "OPTICS";
 }
 return "UNKNOWN VALUE";
}
static inline bool tpuv4common_unused_register_two_orientation_value_is_valid(int value)
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
 if (value == 3) {
  return true;
 }
 return false;
}
static inline const char *tpuv4common_unused_register_two_orientation_value_name(
 tpuv4common_unused_register_two_orientation_value value)
{
 if (value == 0) {
  return "UNKNOWN";
 }
 if (value == 1) {
  return "X";
 }
 if (value == 2) {
  return "Y";
 }
 if (value == 3) {
  return "Z";
 }
 return "UNKNOWN VALUE";
}
static inline bool tpuv4common_unused_register_two_polarity_value_is_valid(int value)
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
static inline const char *tpuv4common_unused_register_two_polarity_value_name(
 tpuv4common_unused_register_two_polarity_value value)
{
 if (value == 0) {
  return "UNKNOWN";
 }
 if (value == 1) {
  return "POSITIVE";
 }
 if (value == 2) {
  return "NEGATIVE";
 }
 return "UNKNOWN VALUE";
}
static inline uint8 tpuv4common_unused_register_two_local_chip(const uint64 reg_value)
{
 return (uint8)((((reg_value >> 0) & 0xffULL) << 0));
}
static inline int set_tpuv4common_unused_register_two_local_chip(uint64 *reg_value,
       uint8 value)
{
 if (value & ~(0xffULL))
  return 1;
 (*reg_value) = ((*reg_value) & ~((0xffULL) << 0)) |
         (((value >> 0) & (0xffULL)) << 0);
 return 0;
}
static inline uint8 tpuv4common_unused_register_two_local_port(const uint64 reg_value)
{
 return (uint8)((((reg_value >> 8) & 0xffULL) << 0));
}
static inline int set_tpuv4common_unused_register_two_local_port(uint64 *reg_value,
       uint8 value)
{
 if (value & ~(0xffULL))
  return 1;
 (*reg_value) = ((*reg_value) & ~((0xffULL) << 8)) |
         (((value >> 0) & (0xffULL)) << 8);
 return 0;
}
static inline tpuv4common_unused_register_two_orientation_value
tpuv4common_unused_register_two_orientation(const uint64 reg_value)
{
 return (tpuv4common_unused_register_two_orientation_value)(
  (((reg_value >> 16) & 0xfULL) << 0));
}
static inline int set_tpuv4common_unused_register_two_orientation(
 uint64 *reg_value, tpuv4common_unused_register_two_orientation_value value)
{
 if (value & ~(0xfULL))
  return 1;
 (*reg_value) = ((*reg_value) & ~((0xfULL) << 16)) |
         (((value >> 0) & (0xfULL)) << 16);
 return 0;
}
static inline tpuv4common_unused_register_two_polarity_value
tpuv4common_unused_register_two_polarity(const uint64 reg_value)
{
 return (tpuv4common_unused_register_two_polarity_value)(
  (((reg_value >> 20) & 0xfULL) << 0));
}
static inline int
set_tpuv4common_unused_register_two_polarity(uint64 *reg_value,
        tpuv4common_unused_register_two_polarity_value value)
{
 if (value & ~(0xfULL))
  return 1;
 (*reg_value) = ((*reg_value) & ~((0xfULL) << 20)) |
         (((value >> 0) & (0xfULL)) << 20);
 return 0;
}
static inline uint8 tpuv4common_unused_register_two_remote_chip(const uint64 reg_value)
{
 return (uint8)((((reg_value >> 24) & 0xffULL) << 0));
}
static inline int set_tpuv4common_unused_register_two_remote_chip(uint64 *reg_value,
        uint8 value)
{
 if (value & ~(0xffULL))
  return 1;
 (*reg_value) = ((*reg_value) & ~((0xffULL) << 24)) |
         (((value >> 0) & (0xffULL)) << 24);
 return 0;
}
static inline uint8 tpuv4common_unused_register_two_remote_port(const uint64 reg_value)
{
 return (uint8)((((reg_value >> 32) & 0xffULL) << 0));
}
static inline int set_tpuv4common_unused_register_two_remote_port(uint64 *reg_value,
        uint8 value)
{
 if (value & ~(0xffULL))
  return 1;
 (*reg_value) = ((*reg_value) & ~((0xffULL) << 32)) |
         (((value >> 0) & (0xffULL)) << 32);
 return 0;
}
static inline tpuv4common_unused_register_two_connector_id_value
tpuv4common_unused_register_two_connector_id(const uint64 reg_value)
{
 return (tpuv4common_unused_register_two_connector_id_value)(
  (((reg_value >> 40) & 0xffULL) << 0));
}
static inline int set_tpuv4common_unused_register_two_connector_id(
 uint64 *reg_value, tpuv4common_unused_register_two_connector_id_value value)
{
 if (value & ~(0xffULL))
  return 1;
 (*reg_value) = ((*reg_value) & ~((0xffULL) << 40)) |
         (((value >> 0) & (0xffULL)) << 40);
 return 0;
}
static inline tpuv4common_unused_register_two_connector_type_value
tpuv4common_unused_register_two_connector_type(const uint64 reg_value)
{
 return (tpuv4common_unused_register_two_connector_type_value)(
  (((reg_value >> 48) & 0xfULL) << 0));
}
static inline int set_tpuv4common_unused_register_two_connector_type(
 uint64 *reg_value, tpuv4common_unused_register_two_connector_type_value value)
{
 if (value & ~(0xfULL))
  return 1;
 (*reg_value) = ((*reg_value) & ~((0xfULL) << 48)) |
         (((value >> 0) & (0xfULL)) << 48);
 return 0;
}
static inline tpuv4common_unused_register_two_config_status_value
tpuv4common_unused_register_two_config_status(const uint64 reg_value)
{
 return (tpuv4common_unused_register_two_config_status_value)(
  (((reg_value >> 52) & 0x7ULL) << 0));
}
static inline int set_tpuv4common_unused_register_two_config_status(
 uint64 *reg_value, tpuv4common_unused_register_two_config_status_value value)
{
 if (value & ~(0x7ULL))
  return 1;
 (*reg_value) = ((*reg_value) & ~((0x7ULL) << 52)) |
         (((value >> 0) & (0x7ULL)) << 52);
 return 0;
}
static inline uint32 tpuv4common_rates_speed_mts(const uint64 reg_value)
{
 return (uint32)((((reg_value >> 32) & 0xffffffffULL) << 0));
}
static inline int set_tpuv4common_rates_speed_mts(uint64 *reg_value, uint32 value)
{
 if (value & ~(0xffffffffULL))
  return 1;
 (*reg_value) = ((*reg_value) & ~((0xffffffffULL) << 32)) |
         (((value >> 0) & (0xffffffffULL)) << 32);
 return 0;
}
static inline uint32 tpuv4common_rates_width(const uint64 reg_value)
{
 return (uint32)((((reg_value >> 0) & 0xffffffffULL) << 0));
}
static inline int set_tpuv4common_rates_width(uint64 *reg_value, uint32 value)
{
 if (value & ~(0xffffffffULL))
  return 1;
 (*reg_value) = ((*reg_value) & ~((0xffffffffULL) << 0)) |
         (((value >> 0) & (0xffffffffULL)) << 0);
 return 0;
}
static inline bool tpuv4common_physical_layer_state_value_value_is_valid(int value)
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
static inline const char *tpuv4common_physical_layer_state_value_value_name(
 tpuv4common_physical_layer_state_value_value value)
{
 if (value == 0) {
  return "DISABLED";
 }
 if (value == 1) {
  return "DOWN";
 }
 if (value == 2) {
  return "UP";
 }
 return "UNKNOWN VALUE";
}
static inline tpuv4common_physical_layer_state_value_value
tpuv4common_physical_layer_state_value(const uint64 reg_value)
{
 return (tpuv4common_physical_layer_state_value_value)(
  (((reg_value >> 0) & 0xffffffffffffffffULL) << 0));
}
static inline int
set_tpuv4common_physical_layer_state_value(uint64 *reg_value,
       tpuv4common_physical_layer_state_value_value value)
{
 if (value & ~(0xffffffffffffffffULL))
  return 1;
 (*reg_value) = ((*reg_value) & ~((0xffffffffffffffffULL) << 0)) |
         (((value >> 0) & (0xffffffffffffffffULL)) << 0);
 return 0;
}
static inline bool
tpuv4common_unused_register_one_cable_type_value_is_valid(int value)
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
 if (value == 3) {
  return true;
 }
 return false;
}
static inline const char *tpuv4common_unused_register_one_cable_type_value_name(
 tpuv4common_unused_register_one_cable_type_value value)
{
 if (value == 0) {
  return "UNKNOWN";
 }
 if (value == 1) {
  return "STRAIGHT";
 }
 if (value == 2) {
  return "Y";
 }
 if (value == 3) {
  return "QUAD";
 }
 return "UNKNOWN VALUE";
}
static inline bool
tpuv4common_unused_register_one_connector_end_value_is_valid(int value)
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
static inline const char *
tpuv4common_unused_register_one_connector_end_value_name(
 tpuv4common_unused_register_one_connector_end_value value)
{
 if (value == 0) {
  return "UNKNOWN";
 }
 if (value == 1) {
  return "END1";
 }
 if (value == 2) {
  return "END2";
 }
 return "UNKNOWN VALUE";
}
static inline tpuv4common_unused_register_one_cable_type_value
tpuv4common_unused_register_one_cable_type(const uint64 reg_value)
{
 return (tpuv4common_unused_register_one_cable_type_value)(
  (((reg_value >> 0) & 0xffULL) << 0));
}
static inline int set_tpuv4common_unused_register_one_cable_type(
 uint64 *reg_value, tpuv4common_unused_register_one_cable_type_value value)
{
 if (value & ~(0xffULL))
  return 1;
 (*reg_value) = ((*reg_value) & ~((0xffULL) << 0)) |
         (((value >> 0) & (0xffULL)) << 0);
 return 0;
}
static inline tpuv4common_unused_register_one_connector_end_value
tpuv4common_unused_register_one_connector_end(const uint64 reg_value)
{
 return (tpuv4common_unused_register_one_connector_end_value)(
  (((reg_value >> 8) & 0xffULL) << 0));
}
static inline int set_tpuv4common_unused_register_one_connector_end(
 uint64 *reg_value,
 tpuv4common_unused_register_one_connector_end_value value)
{
 if (value & ~(0xffULL))
  return 1;
 (*reg_value) = ((*reg_value) & ~((0xffULL) << 8)) |
         (((value >> 0) & (0xffULL)) << 8);
 return 0;
}
static inline uint16
tpuv4common_unused_register_one_port_id(const uint64 reg_value)
{
 return (uint16)((((reg_value >> 16) & 0xffffULL) << 0));
}
static inline int set_tpuv4common_unused_register_one_port_id(uint64 *reg_value,
           uint16 value)
{
 if (value & ~(0xffffULL))
  return 1;
 (*reg_value) = ((*reg_value) & ~((0xffffULL) << 16)) |
         (((value >> 0) & (0xffffULL)) << 16);
 return 0;
}
static inline uint8
tpuv4common_unused_register_one_format_revision(const uint64 reg_value)
{
 return (uint8)((((reg_value >> 32) & 0xffULL) << 0));
}
static inline int
set_tpuv4common_unused_register_one_format_revision(uint64 *reg_value,
        uint8 value)
{
 if (value & ~(0xffULL))
  return 1;
 (*reg_value) = ((*reg_value) & ~((0xffULL) << 32)) |
         (((value >> 0) & (0xffULL)) << 32);
 return 0;
}
static inline uint32
tpuv4common_unused_register_three_oui(const uint64 reg_value)
{
 return (uint32)((((reg_value >> 0) & 0xffffffULL) << 0));
}
static inline int
set_tpuv4common_unused_register_three_oui(uint64 *reg_value, uint32 value)
{
 if (value & ~(0xffffffULL))
  return 1;
 (*reg_value) = ((*reg_value) & ~((0xffffffULL) << 0)) |
         (((value >> 0) & (0xffffffULL)) << 0);
 return 0;
}
static inline uint16
tpuv4common_unused_register_three_revision(const uint64 reg_value)
{
 return (uint16)((((reg_value >> 24) & 0xffffULL) << 0));
}
static inline int
set_tpuv4common_unused_register_three_revision(uint64 *reg_value,
        uint16 value)
{
 if (value & ~(0xffffULL))
  return 1;
 (*reg_value) = ((*reg_value) & ~((0xffffULL) << 24)) |
         (((value >> 0) & (0xffffULL)) << 24);
 return 0;
}
static inline uint64 tpuv4common_unused_register_four_value(const uint64 reg_value)
{
 return (uint64)((((reg_value >> 0) & 0xffffffffffffffffULL) << 0));
}
static inline int set_tpuv4common_unused_register_four_value(uint64 *reg_value,
         uint64 value)
{
 if (value & ~(0xffffffffffffffffULL))
  return 1;
 (*reg_value) = ((*reg_value) & ~((0xffffffffffffffffULL) << 0)) |
         (((value >> 0) & (0xffffffffffffffffULL)) << 0);
 return 0;
}
static inline uint64
tpuv4common_unused_register_five_value(const uint64 reg_value)
{
 return (uint64)((((reg_value >> 0) & 0xffffffffffffffffULL) << 0));
}
static inline int
set_tpuv4common_unused_register_five_value(uint64 *reg_value, uint64 value)
{
 if (value & ~(0xffffffffffffffffULL))
  return 1;
 (*reg_value) = ((*reg_value) & ~((0xffffffffffffffffULL) << 0)) |
         (((value >> 0) & (0xffffffffffffffffULL)) << 0);
 return 0;
}
static inline uint64
tpuv4common_unused_register_six_value(const uint64 reg_value)
{
 return (uint64)((((reg_value >> 0) & 0xffffffffffffffffULL) << 0));
}
static inline int
set_tpuv4common_unused_register_six_value(uint64 *reg_value,
       uint64 value)
{
 if (value & ~(0xffffffffffffffffULL))
  return 1;
 (*reg_value) = ((*reg_value) & ~((0xffffffffffffffffULL) << 0)) |
         (((value >> 0) & (0xffffffffffffffffULL)) << 0);
 return 0;
}
#endif
