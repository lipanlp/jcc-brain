import { ref } from 'vue';
export const settingsOpen = ref(false);
export const settingsSection = ref('vision');
export function openSettings(section = 'vision') {
    settingsSection.value = section;
    settingsOpen.value = true;
}
