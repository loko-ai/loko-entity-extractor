import { Box, Button, Input, Stack, Text, Textarea } from "@chakra-ui/react";
import { useCompositeState } from "ds4biz-core";
import { base_transformer } from "../../config/constants";

export function TransBaseForm({ onSubmit }) {
  const state = useCompositeState({
    name: "",
    description: "",
    bp: JSON.stringify(base_transformer, null, 2),
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
        isInvalid={state.name === ""}
      />

      <Text fontSize="xs">
        Blueprint
        <Box as="span" color="red">
          *
        </Box>
      </Text>
      <Textarea
        isInvalid={state.bp === ""}
        rows={15}
        value={state.bp}
        onChange={(e) => (state.bp = e.target.value)}
      />
      <Button
        onClick={(e) => {
          if (state.name !== "" && state.pb != "") {
            onSubmit(state.name, state.bp);
          }
        }}
      >
        Create
      </Button>
    </Stack>
  );
}
