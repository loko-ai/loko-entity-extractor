import { Button, Flex, HStack, Stack } from "@chakra-ui/react";
import { useCompositeState } from "ds4biz-core";
import { useContext } from "react";
import { CLIENT, StateContext } from "../../config/constants";
import { Transformer } from "./Transformer";
import { TransformCreation } from "./TransformerCreation";

export function Transformers({ transformers }) {
  const state = useCompositeState({ view: "list" });
  const _state = useContext(StateContext);
  switch (state.view) {
    case "list":
      return (
        <Stack w="100%" h="100%" spacing="2rem">
          <HStack>
            <Button onClick={(e) => (state.view = "new")}>
              New transformer
            </Button>
          </HStack>

          <Stack>
            {transformers.map((name) => (
              <Transformer
                //onClick={(e) => (state.view = "model")}
                name={name}
                key={name}
                onDelete={(e) =>
                  CLIENT.transformers[name].delete().then((resp) => {
                    _state.refresh = new Date();
                  })
                }
              />
            ))}
          </Stack>
        </Stack>
      );
    case "new":
      return (
        <Flex w="100vw" h="100vh" p="2rem">
          <TransformCreation onClose={(e) => (state.view = "list")} />
        </Flex>
      );

    default:
      break;
  }
}
