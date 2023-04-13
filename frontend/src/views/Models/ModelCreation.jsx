import {
  Button,
  IconButton,
  Stack,
  Tab,
  TabList,
  TabPanel,
  TabPanels,
  Tabs,
} from "@chakra-ui/react";
import { useContext } from "react";
import { CLIENT, StateContext } from "../../config/constants";
import { BaseForm } from "./BaseForm";
import { RiArrowLeftLine } from "react-icons/ri";

export function ModelCreation({ onClose }) {
  const _state = useContext(StateContext);
  return (
    <Stack w="100%" h="100%" spacing="2rem" color='#A9A9A9'>
    <IconButton
          size="sm"
          w="30px"
          h="30px"
          borderRadius={"full"}
          bg='#222222'
          icon={<RiArrowLeftLine />}
          onClick={onClose}
        />
      <Tabs>
        {/* <TabList>
          <Tab>Base</Tab>
          <Tab>Advanced</Tab>
         <Tab>Manual</Tab>
        </TabList> */}
        <TabPanels>
          {/* <TabPanel>Base</TabPanel>
          <TabPanel>Manual</TabPanel> */}
          <TabPanel>
            <BaseForm
              onSubmit={(name, data) => {
                console.log("Name", name);
                CLIENT[name]
                  .post(data)
                  .then((resp) => (_state.refresh = new Date()));
                onClose();
              }}
            />
          </TabPanel>
        </TabPanels>
      </Tabs>
    </Stack>
  );
}
