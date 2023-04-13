import { Button, Flex, HStack, Stack } from "@chakra-ui/react";
import { useCompositeState } from "ds4biz-core";
import { useContext } from "react";
import { CLIENT, StateContext } from "../../config/constants";
import { Predictor } from "./Predictor";
import { PredictorCreation } from "./PredictorCreation";

export function Predictors({ predictors }) {
  const state = useCompositeState({ view: "list" });
  const _state = useContext(StateContext);
  switch (state.view) {
    case "list":
      return (
        <Stack w="100%" h="100%" spacing="2rem">
          <HStack>
            <Button onClick={(e) => (state.view = "new")}>New predictor</Button>
          </HStack>

          <Stack>
            {predictors.map((name) => (
              <Predictor
                //onClick={(e) => (state.view = "model")}
                name={name}
                key={name}
                onDelete={(e) =>
                  CLIENT.predictors[name].delete().then((resp) => {
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
          <PredictorCreation onClose={(e) => (state.view = "list")} />
        </Flex>
      );

    default:
      break;
  }
}
