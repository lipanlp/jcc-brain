<script setup lang="ts">
import ScoutNotice from '../components/ScoutNotice.vue';
import { hudFrame } from '../composables/hud';
</script>
<template><ScoutNotice :notice="hudFrame?.scout_notice" /></template>
