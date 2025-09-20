import { exec } from "child_process";
import { promisify } from "util";

const execAsync = promisify(exec);

const SETUP_COMMANDS: string[] = [
  'npx create-next-app@latest thermonuclear-app --typescript --tailwind --eslint --app --src-dir --import-alias "@/*"',
  'cd thermonuclear-app',
  'npx shadcn-ui@latest init',
  'npx shadcn-ui@latest add button input card dialog form table toast avatar badge alert progress tabs sheet popover tooltip',
  'npm install framer-motion @tanstack/react-query lucide-react cmdk vaul sonner next-themes @radix-ui/react-icons @radix-ui/react-portal',
  'npm install @mantine/core @mantine/hooks @mantine/charts @tabler/icons-react @react-three/fiber three @react-three/drei',
  'npm install -g @lhci/cli'
];

async function runCommand(command: string) {
  console.log(`\n? ${command}`);
  await execAsync(command, { stdio: "inherit" as any });
}

async function launchThermonuclearApp() {
  for (const command of SETUP_COMMANDS) {
    await runCommand(command);
  }
  console.log("\nThermonuclear UI foundation ready.\n");
}

async function validateThermonuclearApp() {
  await runCommand("npm run build");
  await runCommand("lhci autorun");
  console.log("Lighthouse audit complete (target 100/100 across metrics).");
}

async function main() {
  try {
    await launchThermonuclearApp();
    await validateThermonuclearApp();
  } catch (error) {
    console.error("Thermonuclear workflow failed:", error);
    process.exit(1);
  }
}

main();
