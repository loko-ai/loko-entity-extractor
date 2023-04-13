import { Box, Button, Input, Stack, Text, Textarea } from "@chakra-ui/react";
import { useCompositeState } from "ds4biz-core";
import { base_model } from "../../config/constants";

export function BaseForm({ onSubmit }) {
  const state = useCompositeState({
    name: "",
    bp: JSON.stringify(base_model, null, 2),
  });
  return (
    <Stack>
      <Text fontSize="xs">
        Name
        <Box as="span" color="red">
          *
        </Box>
      </Text>
      <Input
        value={state.name}
        onChange={(e) => (state.name = e.target.value)}
        type="text"
      />

      <Text fontSize="xs">
        Blueprint
        <Box as="span" color="red">
          *
        </Box>
      </Text>
      <Textarea
        rows={15}
        value={state.bp}
        onChange={(e) => (state.bp = e.target.value)}
      />
      <Button bg="#222222" onClick={(e) => onSubmit(state.name, state.bp)}>Create</Button>
    </Stack>
  );
}
