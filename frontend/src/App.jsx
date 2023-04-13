import {
  Box,
  Button,
  Flex,
  Heading,
  HStack,
  VStack,
  Image,
  Link,
  Spacer,
  Stack,
  Tab,
  TabList,
  TabPanel,
  TabPanels,
  Tabs,
} from "@chakra-ui/react";
import { useCompositeState } from "ds4biz-core";
import { useEffect, useState } from "react";
import reactLogo from "./assets/react.svg";
import { CLIENT, StateContext } from "./config/constants";
import { Transformers } from "./views/Transformers/Transformers";
import { Models } from "./views/Models/Models";
import { Predictors } from "./views/Predictors/Predictors";
import icon from "./assets/images/favicon.ico";
import {extendTheme, ChakraProvider} from "@chakra-ui/react"
import {theme} from "./theme";


function App() {
  const state = useCompositeState({
    models: [],
    view: "list",
    refresh: null,
  });

  useEffect(() => {
    CLIENT
      .get()
      .then((resp) => (state.models = resp.data))
      .catch((err) => console.log(err));
  }, [state.refresh]);

  switch (state.view) {
    case "list":
      return (
        <StateContext.Provider value={state}>
        <ChakraProvider theme={theme}>
          <Flex w="100vw" h="100vh" bg='#171717'>
          <VStack spacing={4} w="100vw" px={20} pt={5} pl={5}>
          <Box w="93%" px={0}>
          <HStack>
                    <Link href="https://loko-ai.com/" target="_blank">
              <Image mr="0.5rem" w="50px" src={icon}/>
            </Link>
            <Heading size="md" py="2rem" color='#A9A9A9'>
              Loko - NER
            </Heading>
            </HStack>
            </Box>
            <Tabs w="90%" p="0rem">
              <TabPanels>
                <TabPanel>
                  <Models models={state.models} />
                </TabPanel>
              </TabPanels>
            </Tabs>
            </VStack>
          </Flex>
          </ChakraProvider>
        </StateContext.Provider>
      );

    case "model":
      return (
        <Flex w="100vw" h="100vh" p="2rem">
          <Box onClick={(e) => (state.view = "list")}>Details</Box>
        </Flex>
      );
    case "model_creation":
      return (
          <ModelCreation onClose={(e) => (state.view = "list")} />
      );
  }
}

export default App;
