<script setup lang="ts">
import SettingsSheet from "./components/SettingsSheet.vue";
import LabelUnitsSheet from "./components/LabelUnitsSheet.vue";
import CalibrateSheet from "./components/CalibrateSheet.vue";
import DatabaseStatsSheet from "./components/DatabaseStatsSheet.vue";
import AppShell from './layout/AppShell.vue';
import ToastHost from './components/ToastHost.vue';
import AlertHost from './components/AlertHost.vue';
import { useRoute } from 'vue-router';
const route = useRoute();
</script>
<template>
  <RouterView v-if="route.meta.floating" />
  <template v-else>
    <AppShell />
    <SettingsSheet />
    <LabelUnitsSheet />
    <CalibrateSheet />
    <DatabaseStatsSheet />
    <ToastHost />
    <AlertHost />
  </template>
</template>
