import {
  Box,
  Button,
  Input,
  Select,
  Stack,
  Text,
  Textarea,
} from "@chakra-ui/react";
import { useCompositeState } from "ds4biz-core";
import { useContext } from "react";
import { base_transformer, StateContext } from "../../config/constants";

export function PredBaseForm({ onSubmit }) {
  const state = useCompositeState({
    name: "",
    description: "",
  });
  const _state = useContext(StateContext);
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
      <Text fontSize="xs">Description</Text>
      <Textarea
        value={state.description}
        onChange={(e) => (state.description = e.target.value)}
      />

      <Text fontSize="xs">Model ID</Text>
      <Select
        value={state.model_id}
        onChange={(e) => (state.model_id = e.target.value)}
      >
        <option></option>

        {_state.models.map((el) => (
          <option key={el}>{el}</option>
        ))}
      </Select>
      <Text fontSize="xs">Transformer ID</Text>
      <Select
        value={state.transformer_id}
        onChange={(e) => (state.transformer_id = e.target.value)}
      >
        <option></option>
        {_state.transformers.map((el) => (
          <option key={el}>{el}</option>
        ))}
      </Select>
      <Button
        onClick={(e) => {
          onSubmit(
            state.name,
            state.description,
            state.model_id,
            state.transformer_id
          );
        }}
      >
        Create
      </Button>
    </Stack>
  );
}
